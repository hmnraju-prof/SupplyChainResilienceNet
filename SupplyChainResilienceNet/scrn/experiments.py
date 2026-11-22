"""Ablation and sensitivity experiments for manuscript Section 5."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import GraphSequence
from .training import train_model,run_inference,temporal_splits
from .scoring import ResilienceScorer
from .optimization import optimize_recovery


def modified_sequence(data,ablation):
    x=data.x.copy();active=data.edge_active.copy() if data.edge_active is not None else np.ones((data.T,data.E),bool)
    if ablation=="without_maritime":x[:,:,6:10]=0
    elif ablation=="without_historical_disruption":x[:,:,10:14]=0
    elif ablation=="without_propagation":active[:,data.edge_type==5]=False
    elif ablation not in ("full","graph_mean","without_graph","without_temporal","temporal_gru"):
        raise ValueError(f"Unknown ablation {ablation}")
    return GraphSequence(x=x,edge_index=data.edge_index,edge_type=data.edge_type,
                         edge_attr=data.edge_attr,node_type=data.node_type,labels=data.labels,
                         dates=data.dates,node_ids=data.node_ids,
                         manifest={**data.manifest,"ablation":ablation},edge_active=active).validate()


def run_ablations(data,cfg,out,seeds=(11,),variants=None):
    variants=variants or ("full","without_maritime","without_historical_disruption",
                           "without_propagation","graph_mean","without_graph",
                           "without_temporal","temporal_gru")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tr,va,te=temporal_splits(data.T,cfg.window)[0]
    result=[]
    for variant in variants:
        d=modified_sequence(data,variant)
        graph_mode=("mean" if variant=="graph_mean" else "identity" if variant=="without_graph" else "attention")
        temporal_mode=("none" if variant=="without_temporal" else "gru" if variant=="temporal_gru" else "transformer")
        for seed in seeds:
            print(f"Ablation {variant}, seed {seed}",flush=True)
            m=train_model(d,cfg,tr,va,seed,graph_mode,temporal_mode,verbose=False)
            _,_,scores=run_inference(d,cfg,m.model,m.criterion,te)
            result.append({"variant":variant,"seed":seed,**scores})
            pd.DataFrame(result).to_csv(out/"ablations.csv",index=False)
    return pd.DataFrame(result)


def run_threshold_sensitivity(data,cfg,pred,indices,out):
    """Cheap, no-retraining threshold/resilience sensitivity on held-out predictions."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    train,_,_=temporal_splits(data.T,cfg.window)[0]
    scorer=ResilienceScorer.fit(data.labels["delay"][train],data.labels["loss"][train],cfg.ahp_weights)
    N=data.N
    last={k:(v[-N:] if k not in ("type_prob","severity_prob") else v[-N:,:]) for k,v in pred.items()}
    risk=last["risk"];severity=last["severity_prob"];delay=last["delay"];loss=last["loss"]
    nodes=np.array(data.node_ids)
    rows=[]
    for rt in (.35,.5,.65):
        for rs in (.35,.45,.55):
            score=scorer.score(risk,severity,delay,loss)
            tbl=pd.DataFrame({"node_id":nodes})
            tbl["node_type"]=[("supplier","factory","warehouse","port","logistics_hub","demand_region")[i] for i in data.node_type]
            tbl["risk"]=risk;tbl["resilience"]=score;tbl["delay"]=delay;tbl["loss"]=loss
            plan,report=optimize_recovery(tbl,risk_threshold=rt,resilience_threshold=rs)
            rows.append({"risk_threshold":rt,"resilience_threshold":rs,
                         "priority_nodes":report.get("priority_nodes",0),
                         "selected_actions":report.get("actions",0),
                         "expected_cost":report.get("expected_cost",0)})
    pd.DataFrame(rows).to_csv(out/"threshold_sensitivity.csv",index=False)
    return pd.DataFrame(rows)
