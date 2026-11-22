"""Classification, regression and recovery metrics without data fabrication."""
import numpy as np
from sklearn.metrics import (accuracy_score,precision_recall_fscore_support,roc_auc_score,
                             average_precision_score,mean_absolute_error,mean_squared_error,r2_score,confusion_matrix)


def evaluate_outputs(pred, labels,threshold=.5):
    yp=np.asarray(pred["risk"]).reshape(-1)
    yr=np.asarray(labels["risk"]).reshape(-1).astype(int)
    dec=(yp>=threshold).astype(int)
    pp,rr,ff,_=precision_recall_fscore_support(yr,dec,average="binary",zero_division=0)
    metrics={"risk_accuracy":float(accuracy_score(yr,dec)),"risk_precision":float(pp),
             "risk_recall":float(rr),"risk_f1":float(ff),"n_predictions":int(yr.size),
             "positive_rate":float(yr.mean())}
    if len(np.unique(yr))>1:
        metrics["risk_auc"]=float(roc_auc_score(yr,yp))
        metrics["risk_pr_auc"]=float(average_precision_score(yr,yp))
    else:metrics["risk_auc"]=metrics["risk_pr_auc"]=float("nan")
    event=yr==1
    for name,prob in (("type",pred["type_prob"]),("severity",pred["severity_prob"])):
        y=np.asarray(labels[name]).reshape(-1)
        probs=np.asarray(prob).reshape(-1,9 if name=="type" else 4)
        valid=event&(y>=0)
        if valid.any():
            r=precision_recall_fscore_support(y[valid],np.argmax(probs[valid],axis=1),
                                                average="macro",zero_division=0)
            metrics.update({f"{name}_macro_precision":float(r[0]),
                            f"{name}_macro_recall":float(r[1]),
                            f"{name}_macro_f1":float(r[2])})
        else:
            metrics[f"{name}_macro_f1"]=float("nan")
            metrics[f"{name}_macro_precision"]=float("nan")
            metrics[f"{name}_macro_recall"]=float("nan")
    for name in ("delay","loss"):
        valid=event & (np.asarray(labels[f"{name}_valid"]).reshape(-1)>0 if f"{name}_valid" in labels else True)
        true=np.asarray(labels[name]).reshape(-1)[valid]
        est=np.asarray(pred[name]).reshape(-1)[valid]
        if true.size:
            metrics[f"{name}_mae"]=float(mean_absolute_error(true,est))
            metrics[f"{name}_rmse"]=float(np.sqrt(mean_squared_error(true,est)))
            metrics[f"{name}_r2"]=float(r2_score(true,est)) if len(true)>1 else float("nan")
        else:
            for m in ("mae","rmse","r2"):metrics[f"{name}_{m}"]=float("nan")
    return metrics


def concatenate_predictions(predictions):
    if not predictions:raise ValueError("no predictions supplied")
    return {k:np.concatenate([p[k] for p in predictions],axis=0) for k in predictions[0]}


def classification_cm(pred,labels,threshold=.5):
    return confusion_matrix(np.asarray(labels["risk"]).reshape(-1),
                            (np.asarray(pred["risk"]).reshape(-1)>=threshold).astype(int),labels=[0,1])
