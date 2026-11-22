import numpy as np
import pandas as pd
from scrn.scoring import ResilienceScorer,ahp_consistency
from scrn.optimization import optimize_recovery,DEFAULT_CAPACITIES,evaluate_recovery_scenarios


def test_action_independent_resilience():
    score=ResilienceScorer(30,100)
    p=np.array([0.,1.])
    severity=np.array([[1.,0.,0.,0.],[0.,0.,0.,1.]])
    values=score.score(p,severity,np.array([0.,30.]),np.array([0.,100.]))
    assert values[0]>values[1]
    assert np.all((values>=0)&(values<=1))
    assert np.isclose(values[1],0.)


def test_ahp_consistency():
    arr=np.array([[1,2,2,3],[.5,1,1,2],[.5,1,1,2],[1/3,.5,.5,1]])
    result=ahp_consistency(None,arr)
    assert result["CR"]<.1
    assert abs(sum(result["weights"])-1)<1e-8


def test_joint_milp_no_capacity_violations():
    nodes=pd.DataFrame([{"node_id":f"supplier_{j}","node_type":"supplier","risk":.95,
                         "resilience":.20,"delay":12.,"loss":60.} for j in range(8)])
    caps={**DEFAULT_CAPACITIES,"backup_supplier":7.,"recovery_team":4.}
    plan,summary=optimize_recovery(nodes,budget=8,capacities=caps,service_floor=.2)
    assert len(plan)==len(nodes)
    assert summary["direct_recovery_spend"]<=8+1e-6
    for res,cap in caps.items():
        assert plan.loc[plan.resource==res,"resource_usage"].sum()<=cap+1e-5
    assert (plan.groupby("node_id").size()==1).all()
    truth=nodes.rename(columns={"risk":"risk","delay":"delay","loss":"loss"})
    sim=evaluate_recovery_scenarios(plan,truth)
    assert np.isfinite(sim["cost_with_action"])


def test_no_priority_nodes():
    rows=pd.DataFrame([{"node_id":"a","node_type":"port","risk":.1,
                        "resilience":.85,"delay":0,"loss":0}])
    plan,report=optimize_recovery(rows)
    assert plan.empty and report["status"]=="no_priority_nodes"
