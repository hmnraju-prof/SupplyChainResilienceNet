
from dataclasses import dataclass
import numpy as np

@dataclass
class ResilienceScorer:
    delay_max:float
    loss_max:float
    weights:tuple=(.424,.227,.227,.122)

    @classmethod
    def fit(cls,train_delay,train_loss,weights=(.424,.227,.227,.122)):
        # Train-only maxima, no test-set normalisation leakage.
        return cls(max(float(np.max(train_delay)),1.),max(float(np.max(train_loss)),1.),tuple(weights))

    def score(self,risk,severity_prob,delay,loss):
        risk=np.asarray(risk,dtype=float)
        severity_prob=np.asarray(severity_prob,dtype=float)
        assert severity_prob.shape[-1]==4
        sev=np.sum(severity_prob*np.array([.25,.5,.75,1.]),axis=-1)
        d=np.clip(np.asarray(delay,dtype=float)/self.delay_max,0,1)
        c=np.clip(np.asarray(loss,dtype=float)/self.loss_max,0,1)
        w=self.weights
        rs=1-(w[0]*np.clip(risk,0,1)+w[1]*sev+w[2]*d+w[3]*c)
        return np.clip(rs,0,1)

    def priorities(self,risk,severity_prob,delay,loss,risk_cutoff=.5,resilience_cutoff=.45):
        rs=self.score(risk,severity_prob,delay,loss)
        return {"resilience":rs,"priority":(np.asarray(risk)>=risk_cutoff)|(rs<=resilience_cutoff),
                "rank":np.argsort(np.argsort(rs))+1}


def ahp_consistency(weights,comparison):
    comparison=np.asarray(comparison,dtype=float)
    eigval,eigvec=np.linalg.eig(comparison)
    index=np.argmax(eigval.real); dominant=float(eigval[index].real)
    priorities=np.abs(eigvec[:,index].real);priorities/=priorities.sum()
    ci=(dominant-len(priorities))/(len(priorities)-1)
    ri={3:.58,4:.9,5:1.12}.get(len(priorities),1.)
    return {"weights":priorities.tolist(),"lambda_max":dominant,"CI":float(ci),"CR":float(ci/ri)}
