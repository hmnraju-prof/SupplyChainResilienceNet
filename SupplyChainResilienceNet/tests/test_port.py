import numpy as np
import pandas as pd
import pytest
from scrn.port_task import build_port_risk_graph,run_port_task
from scrn.config import Config


def test_real_daily_port_labels_and_small_training(tmp_path):
    rng=np.random.default_rng(42)
    days=pd.date_range("2024-01-01",periods=25)
    df=pd.DataFrame([{"day":d.strftime("%Y-%m-%d"),"port_id":f"P{i}",
                      "turnaround_days":float(1.5+.2*i+.2*np.sin(t/4)+rng.normal(0,.1))}
                     for t,d in enumerate(days) for i in range(3)])
    csv=tmp_path/"daily.csv";df.to_csv(csv,index=False)
    seq=build_port_risk_graph(csv)
    assert seq.T==25 and seq.N==3
    assert seq.manifest["label_nature"]=="observed_turnaround_threshold_derived"
    cfg=Config(hidden=16,graph_layers=1,graph_heads=4,temporal_layers=1,temporal_heads=4,
               feed_forward=32,window=4,epochs=1,patience=1,batch_size=4,report_dpi=100)
    result=run_port_task(csv,cfg,tmp_path/"results")
    assert 0<=result["accuracy"]<=1
    assert (tmp_path/"results"/"port_predictions.csv").exists()


def test_no_semester_upsampling(tmp_path):
    csv=tmp_path/"semiannual.csv"
    pd.DataFrame([{"Economy_Label":"World","period":"2022-S1",
                   "Median_time_in_port_days_Value":1.2}]).to_csv(csv,index=False)
    with pytest.raises(ValueError,match="REQUIRES daily"):build_port_risk_graph(csv)
