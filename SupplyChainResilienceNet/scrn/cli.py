"""One-command demo / preparation / real-data research workflow."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config,NODE_TYPES
from .schema import GraphSequence
from .synthetic import make_demo
from .data import audit_raw,convert_public_sources,load_canonical
from .training import train_model,temporal_splits,run_inference,save_checkpoint,get_device
from .metrics import evaluate_outputs
from .scoring import ResilienceScorer
from .optimization import optimize_recovery,evaluate_recovery_scenarios
from .plots import plot_history,plot_binary,plot_recovery
from .baselines import train_evaluate_tabular
from .statistics import paired_comparison,holm_adjust
from .simulation import apply_recovery_state


def safe_json(obj):
    if isinstance(obj,(np.floating,np.integer)):return obj.item()
    if isinstance(obj,np.ndarray):return obj.tolist()
    if isinstance(obj,Path):return str(obj)
    raise TypeError(type(obj))


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=safe_json) if not _has_nan(obj)
                    else json.dumps(obj,indent=2,default=safe_json),encoding="utf-8")


def _has_nan(o):
    if isinstance(o,float):return not np.isfinite(o)
    if isinstance(o,dict):return any(_has_nan(v) for v in o.values())
    if isinstance(o,(list,tuple)):return any(_has_nan(v) for v in o)
    return False


def build_results_frame(data,cfg,indices,pred,true,score):
    nodes=len(data.node_ids)
    days=np.repeat(np.array(data.dates)[np.array(indices)],nodes)
    ids=np.tile(np.array(data.node_ids),len(indices))
    types=np.tile(np.array([NODE_TYPES[t] for t in data.node_type]),len(indices))
    severity_probs=pred["severity_prob"].reshape(-1,4)
    frames=pd.DataFrame({"day":days,"node_id":ids,"node_type":types,
                         "risk_probability":pred["risk"].reshape(-1),"risk_observed":true["risk"].reshape(-1),
                         "predicted_type":np.argmax(pred["type_prob"].reshape(-1,9),axis=1),
                         "observed_type":true["type"].reshape(-1),
                         "predicted_severity":np.argmax(severity_probs,axis=1),
                         "observed_severity":true["severity"].reshape(-1),
                         "predicted_delay_days":pred["delay"].reshape(-1),
                         "observed_delay_days":true["delay"].reshape(-1),
                         "predicted_loss_kusd":pred["loss"].reshape(-1),
                         "observed_loss_kusd":true["loss"].reshape(-1),
                         "delay_label_available":true.get("delay_valid",np.ones_like(true["risk"])).reshape(-1).astype(int),
                         "loss_label_available":true.get("loss_valid",np.ones_like(true["risk"])).reshape(-1).astype(int),
                         "resilience_score":score.reshape(-1)})
    return frames


def run_workflow(data,cfg,output_dir,seed=11,baselines=False,ablate=False,sensitivity=False):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"figures").mkdir(exist_ok=True)
    data.validate();cfg.validate();cfg.save(out/"configuration.json")
    write_json(out/"data_manifest.json",data.manifest)
    tr,va,te=temporal_splits(data.T,cfg.window,folds=1)[0]
    write_json(out/"split_indices.json",{"train":tr,"validation":va,"test":te,
                         "train_dates":[data.dates[tr[0]],data.dates[tr[-1]]],
                         "validation_dates":[data.dates[va[0]],data.dates[va[-1]]],
                         "test_dates":[data.dates[te[0]],data.dates[te[-1]]]})
    result=train_model(data,cfg,tr,va,seed=seed,verbose=True)
    pred,true,metrics=run_inference(data,cfg,result.model,result.criterion,te)
    scorer=ResilienceScorer.fit(data.labels["delay"][tr],data.labels["loss"][tr],cfg.ahp_weights)
    score=scorer.score(pred["risk"],pred["severity_prob"],pred["delay"],pred["loss"])
    predictions=build_results_frame(data,cfg,te,pred,true,score)
    predictions.to_csv(out/"predictions.csv",index=False)
    daily=predictions[predictions.day==predictions.day.max()].copy()
    decision=daily.rename(columns={"risk_probability":"risk","predicted_delay_days":"delay",
                                     "predicted_loss_kusd":"loss","resilience_score":"resilience"})
    selected,report=optimize_recovery(decision,budget=60.,max_nodes=30,
                                    risk_threshold=cfg.risk_threshold,
                                    resilience_threshold=cfg.resilience_threshold)
    selected.to_csv(out/"recovery_actions.csv",index=False)
    true_last=daily.rename(columns={"risk_observed":"risk","observed_delay_days":"delay",
                                         "observed_loss_kusd":"loss"})
    sim=evaluate_recovery_scenarios(selected,true_last)
    _,state_changes=apply_recovery_state(data,selected,int(te[-1]))
    state_changes.to_csv(out/"simulated_state_changes.csv",index=False)
    plot_history(result.history,out/"figures"/"training_curves.png",cfg.report_dpi)
    plot_binary(pred,true,out/"figures"/"disruption_metrics.png",cfg.report_dpi)
    plot_recovery(sim,out/"figures"/"recovery_simulation.png",cfg.report_dpi)
    pd.DataFrame(result.history).to_csv(out/"training_history.csv",index=False)
    save_checkpoint(out/"model.pt",result,cfg,data)
    summary={"data_source":data.manifest.get("source"),"data_nature":data.manifest.get("label_nature"),
             "graph":{"snapshots":data.T,"nodes":data.N,"edges":data.E,"features":data.F},
             "model_trainable_parameters":int(sum(p.numel() for p in result.model.parameters() if p.requires_grad)),
             "best_epoch":result.best_epoch,"training_seconds":result.elapsed_seconds,
             "metrics":metrics,"recovery_planner":report,"recovery_scenario":sim,
             "reporting_warning":"Metrics originate from this run. Synthetic-demo results are not evidence of real-world accuracy."}
    if sensitivity:
        from .experiments import run_threshold_sensitivity
        run_threshold_sensitivity(data,cfg,pred,te,out/"sensitivity")
    if ablate:
        from .experiments import run_ablations
        run_ablations(data,cfg,out/"ablations",seeds=(seed,))
    if baselines:
        b={}
        for name in ("logistic","random_forest"):
            _,_,b[name]=train_evaluate_tabular(data,tr,va,te,cfg.window,name)
        summary["baselines"]=b
    write_json(out/"results.json",summary)
    print("\nRUN COMPLETE",out.resolve(),"\nMetrics:",json.dumps(metrics,indent=2),flush=True)
    return summary


def run_crossval(data,cfg,out,seeds=(11,23,37,51,73),folds=5,baselines=True,compare_stgcn=False):
    """25 paired runs when 5 seeds x 5 folds; replayable but computationally demanding."""
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for fold,(tr,va,te) in enumerate(temporal_splits(data.T,cfg.window,folds),1):
        for seed in seeds:
            print(f"Fold {fold}/{folds}; seed {seed}",flush=True)
            result=train_model(data,cfg,tr,va,seed=seed,verbose=False)
            _,_,mm=run_inference(data,cfg,result.model,result.criterion,te)
            all_rows.append({"fold":fold,"seed":seed,"model":"SupplyChainResilienceNet",**mm})
            if compare_stgcn:
                st=train_model(data,cfg,tr,va,seed=seed,graph_mode="mean",temporal_mode="gru",verbose=False)
                _,_,st_metrics=run_inference(data,cfg,st.model,st.criterion,te)
                all_rows.append({"fold":fold,"seed":seed,"model":"ST-GCN-info-matched",**st_metrics})
            if baselines:
                for base in ("logistic","random_forest"):
                    _,_,bm=train_evaluate_tabular(data,tr,va,te,cfg.window,base)
                    all_rows.append({"fold":fold,"seed":seed,"model":base,**bm})
            pd.DataFrame(all_rows).to_csv(output/"paired_results.csv",index=False)
    all_df=pd.DataFrame(all_rows)
    comparisons=[]
    baseline_names=(["logistic","random_forest"] if baselines else [])+(["ST-GCN-info-matched"] if compare_stgcn else [])
    for base in baseline_names:
        for metric in ("risk_f1","risk_auc","type_macro_f1","severity_macro_f1","delay_rmse","loss_rmse"):
            a=all_df[all_df.model=="SupplyChainResilienceNet"].sort_values(["fold","seed"])[metric]
            b=all_df[all_df.model==base].sort_values(["fold","seed"])[metric]
            if len(a)>1 and len(a)==len(b) and np.isfinite(a).all() and np.isfinite(b).all():
                comp=paired_comparison(a,b,metric,bootstrap=2000)
                comp["baseline"]=base;comparisons.append(comp)
    if comparisons:
        adjusted=holm_adjust([r["wilcoxon_p_unadjusted"] for r in comparisons])
        for r,p in zip(comparisons,adjusted):r["holm_p"]=p
        pd.DataFrame(comparisons).to_csv(output/"statistical_comparisons.csv",index=False)
    write_json(output/"report.json",{"source":data.manifest,"rows":len(all_rows),
              "statistical_caution":"Rolling-origin samples share data: inferential p values are exploratory.",
              "comparisons":comparisons})
    return all_df


def main(argv=None):
    ap=argparse.ArgumentParser(prog="scrn",description="SupplyChainResilienceNet research implementation")
    sub=ap.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo",help="Generate an explicitly synthetic demo, train, evaluate and optimise")
    demo.add_argument("--config",default="configs/demo.yaml")
    demo.add_argument("--out",default="outputs/demo")
    demo.add_argument("--baselines",action="store_true")
    demo.add_argument("--ablate",action="store_true")
    demo.add_argument("--sensitivity",action="store_true")
    audit=sub.add_parser("audit",help="Inventory raw data files and their CSV schemas")
    audit.add_argument("--raw-root",default="data/raw");audit.add_argument("--out",default="outputs/raw_audit.json")
    raw=sub.add_parser("convert-public",help="Convert downloaded public source files to semi-synthetic canonical form")
    raw.add_argument("--raw-root",default="data/raw");raw.add_argument("--out",default="data/canonical")
    raw.add_argument("--seed",type=int,default=11)
    prepare=sub.add_parser("prepare",help="Validate and build a graph sequence from canonical CSVs")
    prepare.add_argument("--canonical",default="data/canonical");prepare.add_argument("--out",default="data/processed")
    train=sub.add_parser("train",help="Train and evaluate a previously prepared dataset")
    train.add_argument("--input",default="data/processed");train.add_argument("--config",default="configs/paper.yaml")
    train.add_argument("--out",default="outputs/paper");train.add_argument("--baselines",action="store_true")
    train.add_argument("--ablate",action="store_true");train.add_argument("--sensitivity",action="store_true")
    cv=sub.add_parser("crossval",help="Run rolling-origin / multi-seed paired experiments")
    cv.add_argument("--input",default="data/processed");cv.add_argument("--config",default="configs/paper.yaml")
    cv.add_argument("--out",default="outputs/crossval");cv.add_argument("--folds",type=int,default=5)
    cv.add_argument("--seeds",type=int,nargs="+",default=[11,23,37,51,73])
    cv.add_argument("--no-baselines",action="store_true")
    cv.add_argument("--compare-stgcn",action="store_true")
    port=sub.add_parser("port-task",help="Run genuine risk-only forecasting using observed DAILY per-port records")
    port.add_argument("--csv",required=True);port.add_argument("--out",default="outputs/real_port")
    port.add_argument("--config",default="configs/paper.yaml")
    args=ap.parse_args(argv)
    if args.command=="demo":
        cfg=Config.from_yaml(args.config)
        data=make_demo(cfg.demo_days,cfg.demo_nodes_per_type,cfg.seed)
        data.save(Path(args.out)/"generated_data")
        run_workflow(data,cfg,args.out,seed=cfg.seed,baselines=args.baselines,
                     ablate=args.ablate,sensitivity=args.sensitivity)
    elif args.command=="audit":
        report=audit_raw(args.raw_root)
        write_json(args.out,report);print(json.dumps(report,indent=2))
    elif args.command=="convert-public":
        where=convert_public_sources(args.raw_root,args.out,args.seed)
        print("Canonical files generated:",where)
    elif args.command=="prepare":
        data=load_canonical(args.canonical);print("Prepared:",data.save(args.out))
    elif args.command=="train":
        data=GraphSequence.load(args.input);cfg=Config.from_yaml(args.config)
        run_workflow(data,cfg,args.out,baselines=args.baselines,
                     ablate=args.ablate,sensitivity=args.sensitivity)
    elif args.command=="crossval":
        data=GraphSequence.load(args.input);cfg=Config.from_yaml(args.config)
        run_crossval(data,cfg,args.out,args.seeds,args.folds,not args.no_baselines,
                     compare_stgcn=args.compare_stgcn)
    elif args.command=="port-task":
        from .port_task import run_port_task
        cfg=Config.from_yaml(args.config)
        print(json.dumps(run_port_task(args.csv,cfg,args.out),indent=2))

if __name__=="__main__":main()
