import json
import numpy as np
import pandas as pd
import pytest
from scrn.synthetic import make_demo
from scrn.schema import GraphSequence
from scrn.data import load_canonical


def test_save_load_roundtrip(tmp_path):
    seq=make_demo(days=18,per_type=3)
    seq.save(tmp_path)
    s2=GraphSequence.load(tmp_path)
    assert s2.E==seq.E and s2.T==seq.T
    np.testing.assert_array_equal(s2.labels["severity"],seq.labels["severity"])


def test_canonical_three_day_history_and_missing_values(tmp_path):
    days=pd.date_range("2024-01-01",periods=20)
    entities=pd.DataFrame([{"entity_id":"s1","entity_type":"supplier","capacity":20},
                           {"entity_id":"f1","entity_type":"factory","capacity":20},
                           {"entity_id":"p1","entity_type":"port","capacity":20}])
    relations=pd.DataFrame([{"source_id":"s1","target_id":"f1","relation_type":"procurement"},
                            {"source_id":"p1","target_id":"f1","relation_type":"shipment"}])
    op=pd.DataFrame([{"entity_id":node,"day":d.strftime("%Y-%m-%d"),"demand":float(t+1),
                      "inventory":10.} for t,d in enumerate(days) for node in ("s1","f1","p1")])
    events=pd.DataFrame([{"entity_id":"s1","start_date":"2024-01-09",
                          "end_date":"2024-01-11","severity":"high",
                          "disruption_type":"supplier failure","delay_days":4,"loss_kusd":10}])
    port=pd.DataFrame([{"entity_id":"p1","period":"2022-S1","turnaround_days":1.5}])
    for name,table in (("entities",entities),("relations",relations),
                       ("operations",op),("events",events),("ports",port)):
        table.to_csv(tmp_path/f"{name}.csv",index=False)
    seq=load_canonical(tmp_path)
    assert seq.T==20 and seq.N==3
    assert seq.labels["risk"][8,0]==1
    assert seq.labels["risk"][7,0]==0
    assert seq.manifest["port_feature_granularity"]=="semiannual_as_of"
    # Event history first available AFTER completion, i.e. Jan 12.
    # Feature index 10 is historical_risk, standardised; before/after must differ.
    assert np.isclose(seq.x[9,0,10],seq.x[10,0,10])
    assert not np.isclose(seq.x[10,0,10],seq.x[11,0,10])
    assert not seq.manifest["true_daily_port_label_task"]


def test_unaligned_event_rejected(tmp_path):
    days=pd.date_range("2024-01-01",periods=20)
    pd.DataFrame([{"entity_id":"n","entity_type":"supplier"},
                  {"entity_id":"f","entity_type":"factory"}]).to_csv(tmp_path/"entities.csv",index=False)
    pd.DataFrame([{"source_id":"n","target_id":"f","relation_type":"procurement"}]).to_csv(tmp_path/"relations.csv",index=False)
    pd.DataFrame([{"day":str(d.date()),"entity_id":id,"demand":1} for d in days for id in ("n","f")]).to_csv(tmp_path/"operations.csv",index=False)
    pd.DataFrame([{"entity_id":"n","start_date":"1999-01-01"}]).to_csv(tmp_path/"events.csv",index=False)
    with pytest.raises(ValueError,match="No event dates"):load_canonical(tmp_path)
