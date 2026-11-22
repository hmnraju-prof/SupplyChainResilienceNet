
from datetime import date,timedelta
import numpy as np
from .config import NODE_TYPES, EDGE_TYPES
from .schema import GraphSequence


def make_demo(days=36, per_type=8, seed=11):
    if days<14 or per_type<3: raise ValueError("demo requires at least 14 days and 3 nodes per class")
    rng=np.random.default_rng(seed)
    N=per_type*6; T=days; F=16
    node_type=np.repeat(np.arange(6,dtype=np.int64),per_type)
    ids=[f"{NODE_TYPES[tp]}_{j:03d}" for tp in range(6) for j in range(per_type)]
    edges=[];kinds=[]; base=[]
    def link(u,v,kind,dep=0.5):
        if u==v:return
        edges.append((u,v));kinds.append(EDGE_TYPES.index(kind))
        base.append([dep, float(rng.uniform(.05,.3)), .0, float(rng.uniform(.5,1.))])
    for j in range(per_type):
        # supplier -> factory -> warehouse -> demand, maritime port -> hub -> warehouse
        link(j,per_type+j,"procurement",.65)
        link(per_type+j,2*per_type+j,"production",.75)
        link(3*per_type+j,4*per_type+j,"shipment",.6)
        link(4*per_type+j,2*per_type+j,"shipment",.55)
        link(2*per_type+j,5*per_type+j,"inventory",.6)
        link(j,per_type+(j+1)%per_type,"recovery",.25)
        link(per_type+j,2*per_type+(j+1)%per_type,"propagation",.3)
        link(2*per_type+j,5*per_type+(j+1)%per_type,"propagation",.2)
    # Additional cross-tier dependencies, reproducible and upstream-only.
    for _ in range(per_type*2):
        j=int(rng.integers(per_type));s=int(rng.integers(per_type))
        link(j,per_type+s,"procurement",float(rng.uniform(.12,.6)))
    E=len(edges)
    ei=np.array(edges,dtype=np.int64).T; et=np.array(kinds,dtype=np.int64)
    base=np.array(base,dtype=np.float32)
    x=np.zeros((T,N,F),dtype=np.float32)
    labels={"risk":np.zeros((T,N),dtype=np.float32), "type":np.full((T,N),-1,dtype=np.int64),
            "severity":np.full((T,N),-1,dtype=np.int64), "delay":np.zeros((T,N),dtype=np.float32),
            "loss":np.zeros((T,N),dtype=np.float32)}
    attr=np.tile(base[None,:,:],(T,1,1))
    active=np.ones((T,E),dtype=bool)
    node_bias=rng.normal(0,.4,N)
    for t in range(T):
        season=np.sin(2*np.pi*t/14)
        stress=rng.normal(0,1,N)+0.2*node_bias+0.5*season
        if t>0:
            stress += .38*x[t-1,:,0]
        x[t,:,0]=stress
        x[t,:,1]=np.maximum(0,rng.normal(3,1,N)+.6*stress)
        x[t,:,2]=np.maximum(0,2.5-rng.normal(0,.5,N)-.3*stress)
        x[t,:,3]=np.maximum(0,1.2+.3*stress+rng.normal(0,.15,N))
        x[t,:,4]=np.maximum(0,2.8+rng.normal(0,.6,N))
        x[t,:,5]=np.maximum(0,2.+.3*stress+rng.normal(0,.3,N))
        x[t,:,6]=np.sin(2*np.pi*t/7); x[t,:,7]=np.cos(2*np.pi*t/7)
        x[t,:,8]=np.maximum(0,.2*stress+rng.normal(0,.4,N))
        x[t,:,9]=np.where(node_type==3,1.0,0.0)*np.maximum(0,stress)
        x[t,:,10]=np.where(node_type==3,1.,0.)*rng.uniform(.1,.9,N)
        x[t,:,11]=np.maximum(0,stress*.3+rng.normal(0,.5,N))
        x[t,:,12]=np.maximum(0,1.0+rng.normal(0,.2,N))
        x[t,:,13]=np.where(node_type==0,1.,0.)
        # Only completed/past labels are made available as x inputs.
        x[t,:,14]=(labels["risk"][t-1] if t>0 else 0)
        x[t,:,15]= (labels["severity"][t-1].clip(min=0)/3. if t>0 else 0)
        if t>0:
            prev=labels["risk"][t-1]
            attr[t,:,2]=prev[ei[0]]
            attr[t,:,1]=np.clip(.5*base[:,1]+.3*x[t-1,ei[0],3]+.2*x[t-1,ei[1],8],0,1)
        # The probability of y[t] depends on x[t-1] (forecast), NOT x[t].
        if t>0:
            prior=x[t-1,:,0]; priorlag=x[t-1,:,3]
            incoming=np.zeros(N,dtype=np.float32)
            np.add.at(incoming,ei[1],np.maximum(0,prior[ei[0]])*base[:,0])
            incoming=np.clip(incoming/3,0,2)
            logits=-2.0+0.7*prior+0.35*priorlag+0.35*incoming+node_bias*.2
            p=1/(1+np.exp(-logits))
            event=(rng.uniform(size=N)<p)
            labels["risk"][t]=event.astype(np.float32)
            for j in np.flatnonzero(event):
                kind=int((node_type[j]+rng.integers(0,9))%9)
                sev=int(np.clip(np.floor(.8+1.4*p[j]+rng.exponential(.6)),0,3))
                labels["type"][t,j]=kind;labels["severity"][t,j]=sev
                labels["delay"][t,j]=float(np.clip(1+sev*3.5+2*max(prior[j],0)+rng.normal(0,1.2),.1,45))
                labels["loss"][t,j]=float(np.clip(3+sev*11+priorlag[j]*4+rng.normal(0,3),.1,120))
    dates=[(date(2024,1,1)+timedelta(days=t)).isoformat() for t in range(T)]
    return GraphSequence(x=x,edge_index=ei,edge_type=et,edge_attr=attr,node_type=node_type,
                         labels=labels,dates=dates,node_ids=ids,edge_active=active,
                         manifest={"source":"SYNTHETIC_DEMO_ONLY","seed":seed,"label_nature":"synthetic",
                                   "linkage":"deterministically_generated","no_real_world_claims":True,
                                   "task":"next_day_node_disruption"}).validate()
