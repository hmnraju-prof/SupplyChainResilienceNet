
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,confusion_matrix


def plot_history(history,out,dpi=600):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(5.8,3.7))
    x=[row["epoch"] for row in history]
    ax.plot(x,[row["train_loss"] for row in history],label="Train")
    ax.plot(x,[row["val_loss"] for row in history],label="Validation")
    ax.set(xlabel="Training epoch",ylabel="Multi-task loss")
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(out,dpi=dpi,bbox_inches="tight");plt.close(fig)


def plot_binary(pred,true,out,dpi=600):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    y=np.asarray(true["risk"]).ravel().astype(int)
    p=np.asarray(pred["risk"]).ravel()
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    cm=confusion_matrix(y,p>=.5,labels=[0,1])
    m=axs[0].imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):axs[0].text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=11)
    axs[0].set(xlabel="Predicted disruption",ylabel="Observed disruption",xticks=[0,1],yticks=[0,1])
    if len(np.unique(y))==2:
        fpr,tpr,_=roc_curve(y,p)
        axs[1].plot(fpr,tpr,label=f"AUC = {auc(fpr,tpr):.3f}")
    axs[1].plot([0,1],[0,1],linestyle="--",linewidth=1)
    axs[1].set(xlabel="False positive rate",ylabel="True positive rate",xlim=(0,1),ylim=(0,1))
    if len(np.unique(y))==2:
        axs[1].legend(frameon=False)
    fig.tight_layout();fig.savefig(out,dpi=dpi,bbox_inches="tight");plt.close(fig)


def plot_recovery(summary,out,dpi=600):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(4.8,3.8))
    values=[summary.get("cost_without_action",0),summary.get("cost_with_action",0)]
    ax.bar(["No recovery","Selected recovery"],values)
    ax.set_ylabel("Simulated realised disruption cost (arbitrary units)")
    fig.tight_layout();fig.savefig(out,dpi=dpi,bbox_inches="tight");plt.close(fig)
