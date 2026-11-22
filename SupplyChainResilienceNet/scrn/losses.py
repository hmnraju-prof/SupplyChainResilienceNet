"""Leakage-safe masked multi-task objective and class-weight fitting."""
import numpy as np
import torch
import torch.nn.functional as F


def class_weights(y,classes:int):
    y=np.asarray(y).reshape(-1)
    counts=np.bincount(y[(y>=0)&(y<classes)].astype(int),minlength=classes)
    weight=1.0/np.sqrt(np.maximum(1,counts))
    return (weight/max(weight.mean(),1e-8)).astype(np.float32)

class MultiTaskLoss:
    def __init__(self,labels:dict,train_ids,task_weights=(1.,.7,.7,.5,.5),huber_delta=1.):
        train_ids=list(train_ids)
        risk=labels["risk"][train_ids]
        self.pos_weight=torch.tensor([float(np.sqrt(max(1,(risk==0).sum())/max(1,(risk==1).sum())))])
        self.type_weights=torch.tensor(class_weights(labels["type"][train_ids],9))
        self.sev_weights=torch.tensor(class_weights(labels["severity"][train_ids],4))
        self.weights=task_weights;self.delta=huber_delta
        # Fixed, TRAIN-ONLY scaling of positive event impacts.
        delay_valid=(risk>0) & (labels["delay_valid"][train_ids]>0 if "delay_valid" in labels else True)
        loss_valid=(risk>0) & (labels["loss_valid"][train_ids]>0 if "loss_valid" in labels else True)
        self.delay_scale=float(max(1,np.percentile(labels["delay"][train_ids][delay_valid],90))) if delay_valid.any() else 1.
        self.loss_scale=float(max(1,np.percentile(labels["loss"][train_ids][loss_valid],90))) if loss_valid.any() else 1.

    @property
    def scales(self):return (self.delay_scale,self.loss_scale)

    def __call__(self,pred,labels:dict,target:int):
        y=labels["risk"][target]
        w=self.weights
        risk_loss=F.binary_cross_entropy_with_logits(pred["risk_logit"],y,
                  pos_weight=self.pos_weight.to(y.device))
        events=y>0.5
        if bool(events.any()):
            zero=pred["type_logit"].sum()*0
            type_mask=events & (labels["type"][target]>=0)
            sev_mask=events & (labels["severity"][target]>=0)
            delay_mask=events & (labels["delay_valid"][target].bool() if "delay_valid" in labels else events)
            loss_mask=events & (labels["loss_valid"][target].bool() if "loss_valid" in labels else events)
            type_loss=F.cross_entropy(pred["type_logit"][type_mask],labels["type"][target][type_mask],
                                      weight=self.type_weights.to(y.device)) if bool(type_mask.any()) else zero
            sev_loss=F.cross_entropy(pred["severity_logit"][sev_mask],labels["severity"][target][sev_mask],
                                     weight=self.sev_weights.to(y.device)) if bool(sev_mask.any()) else zero
            delay_loss=F.huber_loss(pred["delay"][delay_mask],labels["delay"][target][delay_mask]/self.delay_scale,
                                    delta=self.delta) if bool(delay_mask.any()) else zero
            fin_loss=F.huber_loss(pred["loss"][loss_mask],labels["loss"][target][loss_mask]/self.loss_scale,
                                  delta=self.delta) if bool(loss_mask.any()) else zero
        else:
            # Zero with gradient connection, so all-missing event heads remain safe.
            zero=pred["type_logit"].sum()*0
            type_loss=sev_loss=delay_loss=fin_loss=zero
        total=sum(v*z for v,z in zip(w,(risk_loss,type_loss,sev_loss,delay_loss,fin_loss)))
        return total,{"risk":risk_loss.detach().item(),"type":type_loss.detach().item(),
                      "severity":sev_loss.detach().item(),"delay":delay_loss.detach().item(),
                      "loss":fin_loss.detach().item()}
