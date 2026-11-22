
from pathlib import Path
import json,copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             roc_auc_score,average_precision_score)
from .schema import GraphSequence
from .config import Config
from .model import SupplyChainResilienceNet
from .training import temporal_splits,set_seed,get_device


def build_port_risk_graph(csv_path,seed=11,quantile=.8):
    df=pd.read_csv(csv_path,low_memory=False)
    df.columns=[str(x).lower().strip().replace(" ","_") for x in df.columns]
    date=next((c for c in ("day","date","observation_date","call_date") if c in df),None)
    idcol=next((c for c in ("port_id","port_name","un_locode","unlocode") if c in df),None)
    delay=next((c for c in ("turnaround_days","median_time_in_port_days_value","time_in_port_days","port_stay_days") if c in df),None)
    if not date or not idcol or not delay:
        raise ValueError("Real-port task REQUIRES daily port_ID/date/turnaround_days columns. "
                         "Country-by-semester statistics cannot support this experiment.")
    df[date]=pd.to_datetime(df[date],errors="coerce")
    df[delay]=pd.to_numeric(df[delay],errors="coerce")
    df=df.dropna(subset=[date,idcol,delay]).copy()
    df["day"]=df[date].dt.strftime("%Y-%m-%d")
    # Require an explicitly measured complete daily panel rather than invent missing labels.
    pivot=df.pivot_table(index="day",columns=idcol,values=delay,aggfunc="median")
    pivot=pivot.dropna(axis=0,how="any").sort_index()
    if len(pivot)<20 or pivot.shape[1]<2:
        raise ValueError("Real-port experiment needs >=20 fully observed daily snapshots and >=2 actual ports; "
                         "do not upsample semiannual records or fill missing outcomes.")
    days=pivot.index.tolist();ids=pivot.columns.astype(str).tolist();T,N=pivot.shape
    cut=max(2,round(.7*T))
    d=pivot.to_numpy(dtype=np.float32)
    thresh=np.quantile(d[:cut],quantile,axis=0)
    labels={"risk":(d>thresh[None,:]).astype(np.float32),
            "type":np.full((T,N),-1,np.int64),"severity":np.full((T,N),-1,np.int64),
            "delay":np.zeros((T,N),np.float32),"loss":np.zeros((T,N),np.float32)}
    active=labels["risk"]>0
    labels["type"][active]=1
    # Quantile-derived ordinal severity is NOT an independently observed severity annotation.
    pct=np.divide(d,np.maximum(thresh[None,:],1e-6))
    s=np.digitize(pct,[1.10,1.25,1.50]).astype(np.int64)
    labels["severity"][active]=s[active]
    labels["delay"]=np.maximum(0,d-np.median(d[:cut],axis=0)[None,:]).astype(np.float32)*labels["risk"]
    feat=np.zeros((T,N,16),dtype=np.float32)
    center=np.median(d[:cut],axis=0);spread=np.std(d[:cut],axis=0)+1e-5
    feat[:,:,0]=(d-center)/spread
    feat[:,:,1]=np.concatenate([np.zeros((1,N)),np.diff(d,axis=0)],axis=0)/spread
    feat[:,:,2]=d/np.maximum(1.,center)
    feat[:,:,3]=np.tile(np.std(d[:cut],axis=0),(T,1))
    for col,fidx in (("vessel_calls",4),("cargo_capacity",5),("port_efficiency",6)):
        if col not in df:continue
        panel=df.pivot_table(index="day",columns=idcol,values=col,aggfunc="median")
        panel=panel.reindex(index=days,columns=ids)
        values=panel.to_numpy(dtype=np.float32)
        med=np.nanmedian(values[:cut],axis=0);med=np.nan_to_num(med,nan=0.)
        std=np.nanstd(values[:cut],axis=0)+1e-5
        feat[:,:,fidx]=np.nan_to_num((values-med)/std,nan=0.)
    for t in range(T):
        dt=pd.Timestamp(days[t]).dayofweek
        feat[t,:,7]=np.sin(2*np.pi*dt/7)
        feat[t,:,8]=np.cos(2*np.pi*dt/7)
        # Prior observed exceedance is legal, today's outcome is not used to predict today's label.
        feat[t,:,9]=labels["risk"][t-1] if t>0 else 0
    # Infer statistical similarities using TRAIN ONLY; these are not physical shipping routes.
    corr=np.corrcoef(d[:cut].T)
    corr=np.nan_to_num(corr)
    k=min(3,N-1)
    edges=[];strengths=[]
    for i in range(N):
        near=np.argsort(-np.abs(corr[i]))
        near=[int(j) for j in near if j!=i][:k]
        for j in near:
            edges.append((j,i));strengths.append(float(np.clip(abs(corr[i,j]),0,1)))
    ei=np.array(edges,dtype=np.int64).T;E=len(edges)
    ea=np.zeros((T,E,4),np.float32)
    ea[:,:,0]=np.array(strengths)[None,:]
    ea[:,:,3]=1.
    for t in range(1,T):ea[t,:,2]=labels["risk"][t-1,ei[0]]
    return GraphSequence(x=feat,edge_index=ei,edge_type=np.full(E,1,np.int64),edge_attr=ea,
        node_type=np.full(N,3,np.int64),labels=labels,dates=days,node_ids=ids,
        manifest={"source":"REAL_DAILY_PORT_OBSERVATIONS","label_nature":"observed_turnaround_threshold_derived",
                  "risk_quantile_train_only":float(quantile),"risk_thresholds_by_port":dict(zip(ids,thresh.tolist())),
                  "physical_routes_observed":False,"graph_edges":"train_only_port_turnaround_similarity",
                  "type_and_severity_independently_observed":False,"delay_regression_evaluated":False,
                  "financial_loss_labels_available":False,"task":"binary_next_day_port_congestion_proxy"}).validate()


def run_port_task(csv_path,cfg:Config,out_dir,seed=11):
    from .plots import plot_history,plot_binary
    from .cli import write_json
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    seq=build_port_risk_graph(csv_path,seed)
    tr,va,te=temporal_splits(seq.T,cfg.window)[0]
    set_seed(seed);device=get_device(cfg.device);tn=seq.as_torch(device)
    model=SupplyChainResilienceNet(seq.F,hidden=cfg.hidden,graph_layers=cfg.graph_layers,
                    graph_heads=cfg.graph_heads,temporal_layers=cfg.temporal_layers,
                    temporal_heads=cfg.temporal_heads,feed_forward=cfg.feed_forward,dropout=cfg.dropout).to(device)
    yy=seq.labels["risk"][tr]
    posweight=torch.tensor(float(np.sqrt(max(1,(yy==0).sum())/max(1,(yy==1).sum()))),device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best=np.inf;best_state=None;stale=0;history=[]
    def infer(t):
        x,ea,active=tn.window(t,cfg.window)
        return model(x,tn.edge_index,tn.edge_type,ea,tn.node_type,active)
    for epoch in range(cfg.epochs):
        model.train();losses=[];opt.zero_grad(set_to_none=True)
        order=np.random.permutation(tr)
        for j,t in enumerate(order):
            raw=infer(int(t)); y=tn.labels["risk"][t]
            l=F.binary_cross_entropy_with_logits(raw["risk_logit"],y,pos_weight=posweight)
            losses.append(float(l.detach()))
            bs=min(cfg.batch_size,len(order)-(j//cfg.batch_size)*cfg.batch_size)
            (l/bs).backward()
            if (j+1)%cfg.batch_size==0 or j+1==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                opt.step();opt.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            val=np.mean([float(F.binary_cross_entropy_with_logits(infer(t)["risk_logit"],
                            tn.labels["risk"][t],pos_weight=posweight)) for t in va])
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"val_loss":float(val)})
        if val<best-1e-6:
            best=val;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=cfg.patience:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        probs=np.concatenate([torch.sigmoid(infer(t)["risk_logit"]).cpu().numpy() for t in te])
    y=seq.labels["risk"][te].reshape(-1).astype(int)
    decision=(probs>=cfg.risk_threshold).astype(int)
    metrics={"accuracy":float(accuracy_score(y,decision)),"precision":float(precision_score(y,decision,zero_division=0)),
             "recall":float(recall_score(y,decision,zero_division=0)),"F1":float(f1_score(y,decision,zero_division=0)),
             "AUC":float(roc_auc_score(y,probs)) if len(set(y))>1 else None,
             "PR_AUC":float(average_precision_score(y,probs)) if len(set(y))>1 else None,
             "n_port_day_observations":len(y),"threshold_source":"training_data_only"}
    pd.DataFrame({"day":np.repeat(np.array(seq.dates)[te],seq.N),
                  "port_id":np.tile(seq.node_ids,len(te)),"observed_label":y,
                  "predicted_probability":probs}).to_csv(out/"port_predictions.csv",index=False)
    write_json(out/"port_results.json",{"metrics":metrics,"manifest":seq.manifest})
    plot_history(history,out/"port_history.png",cfg.report_dpi)
    plot_binary({"risk":probs},{"risk":y},out/"port_disruption_metrics.png",cfg.report_dpi)
    torch.save({"model":model.cpu().state_dict(),"config":cfg.json(),"feature_dim":seq.F},out/"port_model.pt")
    return metrics
