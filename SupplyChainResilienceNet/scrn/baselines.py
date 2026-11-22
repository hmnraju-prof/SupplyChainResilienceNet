"""Information-matched tabular baselines using as-of graph and operational features."""
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier,DummyRegressor
from .metrics import evaluate_outputs


def feature_rows(data,targets,window):
    """Features include x[t-1], historic mean/trend and incoming graph exposure.

    These are accessible to all methods at forecast origin t-1. Never read y[t].
    """
    N=data.N;D=data.F; result=[]
    src,dst=data.edge_index
    for t in targets:
        assert t>=window
        w=data.x[t-window:t]
        last=w[-1];means=w.mean(axis=0);trend=w[-1]-w[0]
        agg=np.zeros((N,D),dtype=np.float32)
        denom=np.zeros((N,1),dtype=np.float32)
        weights=data.edge_attr[t-1,:,0]
        np.add.at(agg,dst,last[src]*weights[:,None])
        np.add.at(denom,dst,weights[:,None])
        agg/=np.maximum(1e-4,denom)
        z=np.concatenate([last,means,trend,agg,
                          np.eye(6,dtype=np.float32)[data.node_type]],axis=-1)
        result.append(z)
    return np.concatenate(result,axis=0)


def _train_clf(X,y,algorithm,classes):
    if len(np.unique(y))<2:
        return DummyClassifier(strategy="constant",constant=int(y[0])).fit(X,y)
    if algorithm=="logistic":return LogisticRegression(max_iter=300,class_weight="balanced").fit(X,y)
    if algorithm=="random_forest":return RandomForestClassifier(n_estimators=100,max_depth=12,
                                                        class_weight="balanced",random_state=11,n_jobs=-1).fit(X,y)
    if algorithm=="hist_gradient_boosting":return HistGradientBoostingClassifier(max_iter=100,random_state=11).fit(X,y)
    raise ValueError(f"unrecognised tabular algorithm {algorithm}")


def _train_reg(X,y,algorithm):
    if len(y)<2:return DummyRegressor(strategy="mean").fit(X,y)
    if algorithm=="random_forest":return RandomForestRegressor(n_estimators=100,max_depth=12,random_state=11,n_jobs=-1).fit(X,y)
    return Ridge(alpha=1.0).fit(X,y)


def train_evaluate_tabular(data,train,val,test,window=7,algorithm="logistic"):
    X=feature_rows(data,train,window);V=feature_rows(data,val,window);Z=feature_rows(data,test,window)
    y={name:data.labels[name][train].reshape(-1) for name in data.labels}
    ground={name:data.labels[name][test].reshape(-1) for name in data.labels}
    classifier=_train_clf(X,y["risk"].astype(int),algorithm,2)
    def proba(clf,z,k):
        result=np.zeros((len(z),k),dtype=float)
        p=clf.predict_proba(z)
        result[:,clf.classes_.astype(int)]=p
        return result
    risk=proba(classifier,Z,2)[:,1]
    outputs={"risk":risk}
    mask=y["risk"]==1
    for name,k in (("type",9),("severity",4)):
        if mask.any():
            sub=_train_clf(X[mask],y[name][mask].astype(int),algorithm,k)
            outputs[f"{name}_prob"]=proba(sub,Z,k)
        else:
            outputs[f"{name}_prob"]=np.ones((len(Z),k))/k
    for name in ("delay","loss"):
        if mask.any():
            reg=_train_reg(X[mask],y[name][mask],algorithm)
            outputs[name]=np.clip(reg.predict(Z),0,None)
        else:outputs[name]=np.zeros(len(Z))
    return outputs,ground,evaluate_outputs(outputs,ground)
