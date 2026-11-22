import numpy as np
from scrn.synthetic import make_demo
from scrn.config import Config
from scrn.training import train_model,temporal_splits,run_inference,save_checkpoint,load_checkpoint


def test_full_train_eval_checkpoint(tmp_path):
    data=make_demo(days=18,per_type=3)
    cfg=Config(hidden=16,graph_layers=1,graph_heads=4,temporal_layers=1,
               temporal_heads=4,feed_forward=32,window=4,epochs=1,batch_size=4,patience=1)
    tr,va,te=temporal_splits(data.T,cfg.window)[0]
    result=train_model(data,cfg,tr,va,verbose=False)
    prediction,truth,metrics=run_inference(data,cfg,result.model,result.criterion,te)
    assert prediction["risk"].shape[0]==len(te)*data.N
    assert 0<=metrics["risk_accuracy"]<=1
    save_checkpoint(tmp_path/"model.pt",result,cfg,data)
    model,loaded,scales=load_checkpoint(tmp_path/"model.pt")
    assert loaded.hidden==cfg.hidden
    assert scales[0]>0 and scales[1]>0
