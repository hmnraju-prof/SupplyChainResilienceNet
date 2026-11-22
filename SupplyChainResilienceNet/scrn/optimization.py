
from __future__ import annotations
from dataclasses import dataclass,asdict
from typing import Sequence
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint

# (node types permitted, fractional mitigation, lead days, direct cost, resource, units needed)
ACTION_RULES={
 "shipment_rerouting":({"port","logistics_hub","warehouse"},.55,3.0,3.0,"alternate_route",6.0),
 "supplier_substitution":({"supplier","factory"},.65,4.0,4.0,"backup_supplier",7.0),
 "production_reallocation":({"factory","warehouse"},.55,3.0,3.5,"spare_factory",6.0),
 "inventory_buffering":({"warehouse","demand_region"},.45,2.0,2.3,"buffer_stock",5.0),
 "recovery_activation":({"supplier","factory","warehouse","port","logistics_hub","demand_region"},
                         .35,4.0,1.7,"recovery_team",2.0),
}
DEFAULT_CAPACITIES={"alternate_route":40.,"backup_supplier":35.,"spare_factory":38.,
                    "buffer_stock":35.,"recovery_team":35.}

@dataclass
class RecoveryOption:
    node_id:str
    action:str
    resource:str
    resource_usage:float
    cost:float
    effective_service:float
    residual_delay:float
    mitigated_fraction:float
    forecast_base_cost:float
    feasible:bool
    reason:str=""


def eligible_options(node:dict,service_floor=.65,lead_limit=5.0):
    """Build resource-bounded actions. A no-action fallback is always admitted;
    service threshold applies to proposed corrective actions, not the no-action state.
    """
    pid=str(node["node_id"]);kind=str(node["node_type"])
    risk=float(node["risk"]);delay=max(0.,float(node["delay"]))
    loss=max(0.,float(node["loss"]))
    baseline=max(0.,risk*(loss+1.8*delay)+6*risk)
    base_service=float(np.clip(1-.055*risk*delay,0,1))
    opts=[RecoveryOption(pid,"none","none",0.,baseline,base_service,delay,0.,baseline,True)]
    for name,(types,effect,lead,action_cost,res,units) in ACTION_RULES.items():
        if kind not in types: continue
        residual_delay=delay*(1-effect)
        service=float(np.clip(1-.055*risk*residual_delay,0,1))
        # No guaranteed improvement; cost effectiveness depends on predicted impact.
        expected=risk*(loss*(1-effect)+1.8*residual_delay)+6*risk*(1-effect)+action_cost
        feasible=(lead<=lead_limit and service>=service_floor)
        reason="" if feasible else ("lead_time" if lead>lead_limit else "service_level")
        opts.append(RecoveryOption(pid,name,res,units,float(expected),service,
                                   float(residual_delay),effect,baseline,feasible,reason))
    return opts


def optimize_recovery(nodes:pd.DataFrame,budget:float=70.,capacities:dict|None=None,
                      risk_threshold=.50,resilience_threshold=.45,
                      service_floor=.65,lead_limit=5.,max_nodes:int=30):
    """Joint MILP with one choice per node, shared resources, global budget and constraints.

    Returns choice, feasibility audit, and predicted cost summary. True outcome evaluation
    is performed independently by evaluate_recovery_scenarios.
    """
    required={"node_id","node_type","risk","resilience","delay","loss"}
    if missing:=required-set(nodes.columns):raise ValueError(f"Missing recovery fields: {missing}")
    priorities=nodes.loc[(nodes.risk>=risk_threshold)|(nodes.resilience<=resilience_threshold)].copy()
    priorities=priorities.sort_values(["resilience","risk"],ascending=[True,False]).head(max_nodes)
    if priorities.empty:
        return pd.DataFrame(columns=["node_id","action","resource","cost"]),{
            "status":"no_priority_nodes","expected_cost":0.,"counterfactual_baseline_cost":0.,"actions":0}
    capabilities=DEFAULT_CAPACITIES.copy();capabilities.update(capacities or {})
    choices=[];by_node={}
    for row in priorities.to_dict("records"):
        opts=eligible_options(row,service_floor,lead_limit)
        if not opts:raise RuntimeError("inconsistent action set")
        for o in opts:
            if o.feasible and (o.resource=="none" or capabilities.get(o.resource,0.)>=o.resource_usage):
                idx=len(choices);choices.append(o);by_node.setdefault(o.node_id,[]).append(idx)
    n=len(choices)
    c=np.array([o.cost for o in choices],dtype=float)
    rows=[];lower=[];upper=[]
    for indices in by_node.values():
        row=np.zeros(n);row[indices]=1;rows.append(row);lower.append(1.);upper.append(1.)
    for resource,capacity in capabilities.items():
        row=np.array([o.resource_usage if o.resource==resource else 0. for o in choices])
        if row.any():rows.append(row);lower.append(0.);upper.append(float(capacity))
    # Budget is defined on DIRECT mitigation expenditure, not predicted disruption loss.
    direct=np.array([ACTION_RULES[o.action][3] if o.action in ACTION_RULES else 0. for o in choices])
    rows.append(direct);lower.append(0.);upper.append(float(budget))
    solution=milp(c,integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
                  constraints=LinearConstraint(np.stack(rows),np.array(lower),np.array(upper)),
                  options={"time_limit":30.0,"mip_rel_gap":.001})
    if solution.x is None:
        raise RuntimeError(f"No MILP recovery solution: {solution.message}")
    chosen=[choices[j] for j in np.flatnonzero(solution.x>.5)]
    result=pd.DataFrame([asdict(o) for o in chosen]).sort_values("node_id").reset_index(drop=True)
    report={"status":"optimal" if solution.status==0 else "feasible_time_limit",
            "expected_cost":float(result.cost.sum()),
            "counterfactual_baseline_cost":float(result.forecast_base_cost.sum()),
            "estimated_cost_reduction_pct":float(100*(1-result.cost.sum()/max(1e-9,result.forecast_base_cost.sum()))),
            "actions":int((result.action!="none").sum()),
            "priority_nodes":int(len(priorities)),
            "direct_recovery_spend":float(sum(ACTION_RULES[a][3] for a in result.action if a in ACTION_RULES)),
            "budget":float(budget),"capacity_assumptions":capabilities,
            "simulation_note":"action effectiveness is assumed, not causally estimated from observed data"}
    return result,report


def evaluate_recovery_scenarios(plan:pd.DataFrame,true_events:pd.DataFrame):
    """Evaluate realized scenario cost with the SAME exogenous outcomes for every policy.

    true_events columns: node_id, risk (0/1 realized), delay, loss.
    No result is claimed as actual field deployment performance.
    """
    if plan.empty:return {"cost_without_action":0.,"cost_with_action":0.,"cost_reduction_pct":0.}
    truth=true_events.set_index("node_id")
    baseline=0.;intervened=0.;service_before=[];service_after=[]
    for record in plan.to_dict("records"):
        if record["node_id"] not in truth.index:continue
        r=truth.loc[record["node_id"]]
        event=float(r.risk);delay=max(0,float(r.delay));loss=max(0,float(r.loss))
        b=event*(loss+1.8*delay+6)
        effect=float(record["mitigated_fraction"])
        direct=ACTION_RULES[record["action"]][3] if record["action"] in ACTION_RULES else 0.
        a=event*(loss*(1-effect)+1.8*delay*(1-effect)+6*(1-effect))+direct
        baseline+=b;intervened+=a
        service_before.append(np.clip(1-.055*event*delay,0,1))
        service_after.append(np.clip(1-.055*event*delay*(1-effect),0,1))
    return {"cost_without_action":float(baseline),"cost_with_action":float(intervened),
            "cost_reduction_pct":float(100*(baseline-intervened)/max(1e-9,baseline)),
            "service_before":float(np.mean(service_before)) if service_before else 1.,
            "service_after":float(np.mean(service_after)) if service_after else 1.,
            "service_level_improvement_pct":float(100*(np.mean(service_after)-np.mean(service_before)) /
                 max(1e-9,np.mean(service_before))) if service_before else 0.}
