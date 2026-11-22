
from __future__ import annotations
import copy,random,time
from dataclasses import dataclass
import numpy as np
import torch
from .config import Config
from .model import SupplyChainResilienceNet
from .losses import MultiTaskLoss
from .metrics import evaluate_outputs,concatenate_predictions


def set_seed(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    # Prefer reproducible CUDA behaviour when possible.
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def get_device(choice="auto"):
    if choice=="auto":return "cuda" if torch.cuda.is_available() else "cpu"
    if choice=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    return choice


def temporal_splits(T,window=7,folds=1):
    """Returns explicit train/validation/test target-day indices. Last fold is 70/15/15.

    Earlier rolling origins keep validation and test length fixed while growing training
    endpoint. Test sets may overlap across folds (documented in README).
    """
    val_len=max(2,round(.15*T));test_len=max(2,T-round(.85*T))
    final_train=T-val_len-test_len
    if final_train < window+3:raise ValueError("not enough days for chronological split")
    endpoints=([final_train] if folds==1 else np.linspace(max(window+3,round(.45*T)),final_train,folds,dtype=int).tolist())
    out=[]
    for end in endpoints:
        train=list(range(window,end));val=list(range(end,end+val_len));test=list(range(end+val_len,end+val_len+test_len))
        if max(test)>=T or not train:raise ValueError("rolling split would exceed available history")
        out.append((train,val,test))
    return out


def _batch_pred(model,seq,target,cfg):
    feats,edgeattr,active=seq.window(target,cfg.window)
    return model(feats,seq.edge_index,seq.edge_type,edgeattr,seq.node_type,active)

@dataclass
class TrainResult:
    model:SupplyChainResilienceNet
    criterion:MultiTaskLoss
    history:list
    best_epoch:int
    best_val:float
    elapsed_seconds:float


def train_model(data,cfg:Config,train_ids,val_ids,seed=None,graph_mode="attention",temporal_mode="transformer",verbose=True):
    seed=cfg.seed if seed is None else int(seed)
    set_seed(seed)
    cfg.validate()
    device=get_device(cfg.device)
    seq=data.as_torch(device)
    model=SupplyChainResilienceNet(data.F,hidden=cfg.hidden,graph_layers=cfg.graph_layers,
              graph_heads=cfg.graph_heads,temporal_layers=cfg.temporal_layers,
              temporal_heads=cfg.temporal_heads,feed_forward=cfg.feed_forward,
              dropout=cfg.dropout,graph_mode=graph_mode,temporal_mode=temporal_mode).to(device)
    objective=MultiTaskLoss(data.labels,train_ids,cfg.lambda_tasks,cfg.loss_huber_delta)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,cfg.epochs))
    best=float("inf");best_epoch=-1;best_weights=None;stale=0;history=[]
    start=time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        order=np.random.permutation(train_ids)
        optimizer.zero_grad(set_to_none=True)
        train_sum=0.
        for j,target in enumerate(order):
            out=_batch_pred(model,seq,int(target),cfg)
            l,_=objective(out,seq.labels,int(target))
            train_sum+=float(l.detach())
            # Last partial batch should have its own correct denominator.
            current_batch=min(cfg.batch_size,len(order)-(j//cfg.batch_size)*cfg.batch_size)
            (l/current_batch).backward()
            if (j+1)%cfg.batch_size==0 or j+1==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            values=[float(objective(_batch_pred(model,seq,int(t),cfg),seq.labels,int(t))[0]) for t in val_ids]
        validation=float(np.mean(values))
        train_mean=train_sum/len(order)
        history.append({"epoch":epoch+1,"train_loss":train_mean,"val_loss":validation})
        if verbose:print(f"  epoch={epoch+1:03d} train={train_mean:.4f} validation={validation:.4f}",flush=True)
        if validation < best-1e-6:
            best=validation;best_epoch=epoch+1;best_weights=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        schedule.step()
        if stale>=cfg.patience:break
    assert best_weights is not None
    model.load_state_dict(best_weights)
    return TrainResult(model,objective,history,best_epoch,best,time.perf_counter()-start)


def run_inference(data,cfg,model,criterion,target_ids,return_dataframe=False):
    device=next(model.parameters()).device
    seq=data.as_torch(device)
    outputs=[];targets=[]
    model.eval()
    with torch.no_grad():
        for t in target_ids:
            raw=_batch_pred(model,seq,int(t),cfg)
            outputs.append(model.predict_numpy(raw,criterion.scales))
            targets.append({k:np.asarray(v[t],dtype=np.float64).copy() for k,v in data.labels.items()})
    pred=concatenate_predictions(outputs)
    true=concatenate_predictions(targets)
    metrics=evaluate_outputs(pred,true,cfg.risk_threshold)
    return pred,true,metrics


def save_checkpoint(path,result:TrainResult,cfg:Config,data):
    from pathlib import Path
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":result.model.cpu().state_dict(),"config":cfg.json(),
               "feature_dim":data.F,"delay_scale":result.criterion.delay_scale,
               "loss_scale":result.criterion.loss_scale,"best_epoch":result.best_epoch,
               "manifest":data.manifest},path)
    return path


def load_checkpoint(path,device="cpu"):
    # weights_only is available in contemporary PyTorch and protects against arbitrary pickle code.
    contents=torch.load(path,map_location="cpu",weights_only=True)
    cfg=Config(**contents["config"])
    model=SupplyChainResilienceNet(contents["feature_dim"],hidden=cfg.hidden,
           graph_layers=cfg.graph_layers,graph_heads=cfg.graph_heads,
           temporal_layers=cfg.temporal_layers,temporal_heads=cfg.temporal_heads,
           feed_forward=cfg.feed_forward,dropout=cfg.dropout)
    model.load_state_dict(contents["model"])
    model.to(device).eval()
    return model,cfg,(contents["delay_scale"],contents["loss_scale"])
