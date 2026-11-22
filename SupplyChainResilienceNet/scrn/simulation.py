"""Explicit hypothetical graph-state update from predicted recovery actions."""
import copy
import numpy as np
import pandas as pd
from .schema import GraphSequence


def apply_recovery_state(data:GraphSequence,action_plan:pd.DataFrame,day_index:int):
    """Return (new_sequence, change_log) with local toy-dynamics in standardised feature units.

    No causal/physical validity is asserted. Source dataset is NOT modified.
    No future target labels are read or changed. Applied delta is a counterfactual scenario.
    """
    if day_index<0 or day_index>=data.T:raise IndexError(day_index)
    altered=GraphSequence(x=data.x.copy(),edge_index=data.edge_index.copy(),
                          edge_type=data.edge_type.copy(),edge_attr=data.edge_attr.copy(),
                          node_type=data.node_type.copy(),labels={k:v.copy() for k,v in data.labels.items()},
                          dates=list(data.dates),node_ids=list(data.node_ids),
                          manifest={**data.manifest,"counterfactual":"hypothetical action effect, not real outcomes"},
                          edge_active=data.edge_active.copy() if data.edge_active is not None else None)
    lookup={v:i for i,v in enumerate(data.node_ids)}
    changes=[]
    for row in action_plan.to_dict("records"):
        nid=row["node_id"];action=row["action"]
        if nid not in lookup or action=="none":continue
        i=lookup[nid]
        effect=float(row.get("mitigated_fraction",.2))
        # Indexed standardised features: 1=production, 2=inventory, 3=delivery,
        # 4=lead time, 5=capacity. No target labels ever altered.
        delta={
            "shipment_rerouting":{4:-.5*effect,3:.25*effect},
            "supplier_substitution":{5:.4*effect,4:-.2*effect},
            "production_reallocation":{1:.5*effect,5:.25*effect},
            "inventory_buffering":{2:.6*effect,3:.25*effect},
            "recovery_activation":{4:-.2*effect,5:.1*effect},
        }.get(action,{})
        for channel,change in delta.items():
            altered.x[day_index,i,channel]+=change
            changes.append({"day":data.dates[day_index],"node_id":nid,"action":action,
                            "feature_index":channel,"standardized_delta":float(change),
                            "interpretation":"simulated_assumed_effect"})
        if action=="shipment_rerouting":
            outgoing=data.edge_index[0]==i
            if outgoing.any():altered.edge_attr[day_index,outgoing,1]*=(1-effect)
    altered.validate()
    return altered,pd.DataFrame(changes,columns=["day","node_id","action","feature_index",
                                                  "standardized_delta","interpretation"])
