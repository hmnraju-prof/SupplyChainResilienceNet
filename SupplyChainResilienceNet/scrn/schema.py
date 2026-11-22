"""Canonical graph-sequence structure and defensive checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from .config import NODE_TYPES, EDGE_TYPES

@dataclass
class GraphSequence:
    # All labels at index t describe the outcome at t, all predictor features at index t describe as-of t.
    x: np.ndarray                    # [T,N,F]
    edge_index: np.ndarray           # [2,E] integer global indices
    edge_type: np.ndarray            # [E] 0..5
    edge_attr: np.ndarray            # [T,E,4] dependency, delay, prior risk, capacity
    node_type: np.ndarray            # [N] 0..5
    labels: dict                     # risk [T,N], type/severity [T,N] (-1 if not applicable), delay/loss [T,N]
    dates: list[str]
    node_ids: list[str]
    manifest: dict
    edge_active: np.ndarray | None = None  # [T,E] optional

    @property
    def T(self): return int(self.x.shape[0])
    @property
    def N(self): return int(self.x.shape[1])
    @property
    def F(self): return int(self.x.shape[2])
    @property
    def E(self): return int(self.edge_index.shape[1])

    def validate(self):
        t,n,f = self.x.shape
        e = self.edge_index.shape[1]
        assert f >= 1 and t >= 4 and n >= 2, "insufficient data"
        assert self.edge_index.shape == (2,e)
        assert self.edge_attr.shape == (t,e,4)
        assert self.edge_type.shape == (e,) and self.node_type.shape == (n,)
        assert len(self.dates) == t and len(self.node_ids) == n
        assert len(set(self.node_ids)) == n, "duplicate entity IDs"
        assert max(self.edge_index.reshape(-1)) < n and min(self.edge_index.reshape(-1)) >= 0
        assert np.max(self.edge_type) < len(EDGE_TYPES) and np.min(self.edge_type) >= 0
        assert np.max(self.node_type) < len(NODE_TYPES) and np.min(self.node_type) >= 0
        assert np.isfinite(self.x).all() and np.isfinite(self.edge_attr).all(), "nonfinite features"
        assert sorted(self.dates) == self.dates and len(set(self.dates)) == t
        for k in ("risk", "type", "severity", "delay", "loss"):
            assert k in self.labels and self.labels[k].shape == (t,n), f"incorrect {k} targets"
        assert np.isin(self.labels["risk"], [0,1]).all(), "binary risk expected"
        pos = self.labels["risk"] == 1
        for k, classes in (("type",9),("severity",4)):
            assert np.all(self.labels[k][~pos] == -1), f"{k} target must be masked for negatives"
            assert np.all((self.labels[k][pos] == -1) | ((self.labels[k][pos] >= 0)&(self.labels[k][pos] < classes)))
        assert np.all(self.labels["delay"] >= 0) and np.all(self.labels["loss"] >= 0)
        for k in ("delay_valid","loss_valid"):
            if k in self.labels:assert self.labels[k].shape==(t,n)
        if self.edge_active is not None:
            assert self.edge_active.shape == (t,e)
        return self

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        arrays = {"x":self.x, "edge_index":self.edge_index, "edge_type":self.edge_type,
                  "edge_attr":self.edge_attr, "node_type":self.node_type}
        arrays.update({f"y_{k}":v for k,v in self.labels.items()})
        if self.edge_active is not None: arrays["edge_active"] = self.edge_active
        np.savez_compressed(directory/"sequence.npz", **arrays)
        meta = {"dates":self.dates,"node_ids":self.node_ids,"manifest":self.manifest,
                "shape":{"days":self.T,"nodes":self.N,"edges":self.E,"features":self.F}}
        (directory/"manifest.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        arr = np.load(directory/"sequence.npz",allow_pickle=False)
        meta = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        label_names=("risk","type","severity","delay","loss","delay_valid","loss_valid")
        return cls(x=arr["x"],edge_index=arr["edge_index"],edge_type=arr["edge_type"],
                   edge_attr=arr["edge_attr"],node_type=arr["node_type"],
                   labels={k:arr[f"y_{k}"] for k in label_names if f"y_{k}" in arr},
                   dates=meta["dates"],node_ids=meta["node_ids"],manifest=meta["manifest"],
                   edge_active=arr["edge_active"] if "edge_active" in arr else None).validate()

    def as_torch(self, device="cpu"):
        def to(arr,dtype): return torch.as_tensor(arr,dtype=dtype,device=device)
        return TorchSequence(to(self.x,torch.float32), to(self.edge_index,torch.long),
                             to(self.edge_type,torch.long),to(self.edge_attr,torch.float32),
                             to(self.node_type,torch.long),
                             {k:to(v,torch.long if k in ("type","severity") else torch.float32) for k,v in self.labels.items()},
                             to(self.edge_active,torch.bool) if self.edge_active is not None else None)

    def to_pyg(self, time_index:int):
        """Optional, exact per-snapshot PyG HeteroData adapter; pure Torch model does not require PyG."""
        try:
            from torch_geometric.data import HeteroData
        except ImportError as exc:
            raise ImportError("Optional PyG export requires: pip install torch_geometric") from exc
        data=HeteroData()
        ids_per_type={}
        for tp, name in enumerate(NODE_TYPES):
            ids=np.flatnonzero(self.node_type==tp)
            ids_per_type[tp]=ids
            data[name].x=torch.as_tensor(self.x[time_index,ids],dtype=torch.float32)
            data[name].global_node_id=torch.as_tensor(ids,dtype=torch.long)
        maps={i:(int(self.node_type[i]),int(np.searchsorted(ids_per_type[self.node_type[i]],i))) for i in range(self.N)}
        store={}
        active=self.edge_active[time_index] if self.edge_active is not None else np.ones(self.E,dtype=bool)
        for idx in np.flatnonzero(active):
            u,v=map(int,self.edge_index[:,idx]); ut,ui=maps[u];vt,vi=maps[v]
            key=(NODE_TYPES[ut],EDGE_TYPES[int(self.edge_type[idx])],NODE_TYPES[vt])
            store.setdefault(key,[]).append((ui,vi,idx))
        for key, records in store.items():
            data[key].edge_index=torch.tensor([[a for a,b,c in records],[b for a,b,c in records]],dtype=torch.long)
            data[key].edge_attr=torch.as_tensor(self.edge_attr[time_index,[c for a,b,c in records]],dtype=torch.float32)
        return data

@dataclass
class TorchSequence:
    x:torch.Tensor
    edge_index:torch.Tensor
    edge_type:torch.Tensor
    edge_attr:torch.Tensor
    node_type:torch.Tensor
    labels:dict
    edge_active:torch.Tensor|None

    @property
    def T(self): return self.x.shape[0]

    def window(self,target:int,length:int):
        if target < length or target>=self.T:
            raise IndexError("target time must have the complete historical context")
        # x[t-k...t-1] always predicts y[t] (one-step ahead)
        return self.x[target-length:target],self.edge_attr[target-length:target],(
            self.edge_active[target-length:target] if self.edge_active is not None else None)
