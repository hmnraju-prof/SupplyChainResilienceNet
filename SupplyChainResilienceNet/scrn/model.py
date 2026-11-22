"""Edge-aware relation-specific GAT + temporal Transformer, implemented in standard PyTorch.

The implementation operates on a canonical typed-graph tensor; `GraphSequence.to_pyg()`
provides an optional PyG HeteroData interoperability layer. We avoid pyg_lib/scatter
binary dependencies and instead use torch's differentiable native scatter operations.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .config import EDGE_TYPES, NODE_TYPES

class RelationalAttention(nn.Module):
    def __init__(self, hidden:int, heads:int=4, edge_dim:int=4, relations:int=6,
                 dropout:float=0.2, aggregate:str="attention"):
        super().__init__()
        if hidden % heads: raise ValueError("hidden must divide by heads")
        self.hidden,self.heads,self.dim = hidden,heads,hidden//heads
        self.aggregate=aggregate
        self.q=nn.Linear(hidden,hidden,bias=False)
        self.k=nn.Linear(hidden,hidden,bias=False)
        self.v=nn.Linear(hidden,hidden,bias=False)
        self.selfproj=nn.Linear(hidden,hidden)
        self.rel_key=nn.Embedding(relations,hidden)
        self.rel_value=nn.Embedding(relations,hidden)
        self.edge_key=nn.Linear(edge_dim,hidden,bias=False)
        self.edge_value=nn.Linear(edge_dim,hidden,bias=False)
        self.attn_bias=nn.Linear(edge_dim,heads,bias=False)
        self.out=nn.Linear(hidden,hidden)
        self.norm=nn.LayerNorm(hidden)
        self.drop=nn.Dropout(dropout)
        self.attn_drop=nn.Dropout(min(.1,dropout))

    def forward(self, h, edge_index, edge_type, edge_attr, edge_active=None):
        n=h.size(0)
        if edge_active is not None:
            edge_index=edge_index[:,edge_active]
            edge_type=edge_type[edge_active]
            edge_attr=edge_attr[edge_active]
        if edge_index.size(1)==0:
            return self.norm(h + self.drop(self.selfproj(h)))
        src,dst=edge_index
        q=self.q(h).reshape(n,self.heads,self.dim)
        k=self.k(h).reshape(n,self.heads,self.dim)
        v=self.v(h).reshape(n,self.heads,self.dim)
        rk=self.rel_key(edge_type).reshape(-1,self.heads,self.dim)
        rv=self.rel_value(edge_type).reshape(-1,self.heads,self.dim)
        ek=self.edge_key(edge_attr).reshape(-1,self.heads,self.dim)
        ev=self.edge_value(edge_attr).reshape(-1,self.heads,self.dim)
        if self.aggregate=="attention":
            scores=F.leaky_relu((q[dst]*(k[src]+rk+ek)).sum(-1)/math.sqrt(self.dim)+
                                self.attn_bias(edge_attr),negative_slope=.2)
            max_node=torch.full((n,self.heads),-torch.inf,dtype=h.dtype,device=h.device)
            max_node=max_node.scatter_reduce(0,dst[:,None].expand_as(scores),scores,
                                             reduce="amax",include_self=True)
            ex=torch.exp(scores-max_node[dst])
            den=torch.zeros((n,self.heads),dtype=h.dtype,device=h.device)
            den=den.index_add(0,dst,ex)
            alpha=self.attn_drop(ex/den[dst].clamp_min(1e-8))
        elif self.aggregate=="mean":
            counts=torch.zeros((n,self.heads),dtype=h.dtype,device=h.device)
            ones=torch.ones((src.numel(),self.heads),dtype=h.dtype,device=h.device)
            counts=counts.index_add(0,dst,ones)
            alpha=ones/counts[dst].clamp_min(1)
        else:
            raise ValueError(f"Unknown relation aggregator: {self.aggregate}")
        messages=(v[src]+rv+ev)*alpha.unsqueeze(-1)
        result=torch.zeros((n,self.heads*self.dim),device=h.device,dtype=h.dtype)
        result=result.index_add(0,dst,messages.reshape(-1,self.hidden))
        return self.norm(h+self.drop(F.gelu(self.out(result)+self.selfproj(h))))


def positional_encoding(length,dim,device,dtype):
    position=torch.arange(length,device=device,dtype=dtype)[:,None]
    divisor=torch.exp(torch.arange(0,dim,2,device=device,dtype=dtype)*(-math.log(10000.0)/dim))
    pe=torch.zeros((length,dim),device=device,dtype=dtype)
    pe[:,0::2]=torch.sin(position*divisor)
    # Supports odd hidden dimensions (although actual config uses even dimension).
    pe[:,1::2]=torch.cos(position*divisor[:pe[:,1::2].shape[1]])
    return pe

class SupplyChainResilienceNet(nn.Module):
    def __init__(self,feature_dim:int,hidden:int=128,graph_layers:int=3,graph_heads:int=4,
                 temporal_layers:int=2,temporal_heads:int=4,feed_forward:int=256,
                 dropout:float=.2,graph_mode:str="attention",temporal_mode:str="transformer"):
        super().__init__()
        self.hidden=hidden;self.graph_mode=graph_mode;self.temporal_mode=temporal_mode
        self.input_proj=nn.Linear(feature_dim,hidden)
        self.type_embed=nn.Embedding(len(NODE_TYPES),hidden)
        self.spatial=nn.ModuleList([
            RelationalAttention(hidden,graph_heads,relations=len(EDGE_TYPES),dropout=dropout,
                                aggregate="mean" if graph_mode=="mean" else "attention")
            for _ in range(graph_layers)])
        if temporal_mode=="transformer":
            unit=nn.TransformerEncoderLayer(d_model=hidden,nhead=temporal_heads,
                                           dim_feedforward=feed_forward,dropout=dropout,
                                           activation="gelu",batch_first=True,norm_first=False)
            self.temporal=nn.TransformerEncoder(unit,num_layers=temporal_layers,enable_nested_tensor=False)
        elif temporal_mode=="gru":
            self.temporal=nn.GRU(hidden,hidden,num_layers=temporal_layers,batch_first=True,
                                 dropout=dropout if temporal_layers>1 else 0.)
        elif temporal_mode!="none":
            raise ValueError("temporal_mode must be transformer, gru, or none")
        self.fusion=nn.Sequential(nn.Linear(2*hidden+feature_dim,256),nn.ReLU(),
                                  nn.Dropout(dropout),nn.Linear(256,hidden),nn.ReLU(),
                                  nn.Dropout(dropout))
        self.risk=nn.Linear(hidden,1)
        self.disruption_type=nn.Linear(hidden,9)
        self.severity=nn.Linear(hidden,4)
        self.delay=nn.Linear(hidden,1)
        self.loss=nn.Linear(hidden,1)
        self.pool_attention=nn.Linear(hidden,1)

    def forward(self,window_x,edge_index,edge_type,window_edge_attr,node_type,window_edge_active=None):
        """Past snapshots [K,N,F] produce next-day predictions [N,...]; no target inputs."""
        k,n,f=window_x.shape
        seq=[]
        for t in range(k):
            x=window_x[t]
            h=F.gelu(self.input_proj(x)+self.type_embed(node_type))
            if self.graph_mode!="identity":
                for layer in self.spatial:
                    h=layer(h,edge_index,edge_type,window_edge_attr[t],
                            window_edge_active[t] if window_edge_active is not None else None)
            seq.append(h)
        full=torch.stack(seq,dim=1) # [nodes, history length, channels]
        if self.temporal_mode=="transformer":
            full=full+positional_encoding(k,self.hidden,full.device,full.dtype)[None,:,:]
            causal=torch.triu(torch.ones(k,k,device=full.device,dtype=torch.bool),diagonal=1)
            history=self.temporal(full,mask=causal)[:,-1,:]
        elif self.temporal_mode=="gru":
            history=self.temporal(full)[0][:,-1,:]
        else:
            history=full[:,-1,:]
        fused=self.fusion(torch.cat((history,seq[-1],window_x[-1]),dim=-1))
        # Attention readout is separate from node-level supervision.
        pool_weights=torch.softmax(self.pool_attention(fused).squeeze(-1),dim=0)
        graph_state=torch.sum(pool_weights[:,None]*fused,dim=0)
        return {"risk_logit":self.risk(fused).squeeze(-1),
                "type_logit":self.disruption_type(fused),
                "severity_logit":self.severity(fused),
                "delay":F.softplus(self.delay(fused).squeeze(-1)),
                "loss":F.softplus(self.loss(fused).squeeze(-1)),
                "graph_state":graph_state}

    @staticmethod
    def predict_numpy(out,scales=(1.,1.)):
        return {"risk":torch.sigmoid(out["risk_logit"]).detach().cpu().numpy(),
                "type_prob":torch.softmax(out["type_logit"],dim=-1).detach().cpu().numpy(),
                "severity_prob":torch.softmax(out["severity_logit"],dim=-1).detach().cpu().numpy(),
                "delay":out["delay"].detach().cpu().numpy()*scales[0],
                "loss":out["loss"].detach().cpu().numpy()*scales[1]}
