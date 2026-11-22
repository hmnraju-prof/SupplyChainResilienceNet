"""Validated settings for SupplyChainResilienceNet."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import yaml

NODE_TYPES = ("supplier", "factory", "warehouse", "port", "logistics_hub", "demand_region")
EDGE_TYPES = ("procurement", "shipment", "production", "inventory", "recovery", "propagation")
DISRUPTION_TYPES = ("supplier_failure", "port_congestion", "transport_delay", "natural_disaster", "geopolitical", "cyberattack", "labour_strike", "factory_incident", "demand_fluctuation")
SEVERITY_TYPES = ("low", "medium", "high", "critical")
ACTIONS = ("none", "shipment_rerouting", "supplier_substitution", "production_reallocation", "inventory_buffering", "recovery_activation")

@dataclass
class Config:
    seed: int = 11
    hidden: int = 128
    graph_layers: int = 3
    graph_heads: int = 4
    temporal_layers: int = 2
    temporal_heads: int = 4
    feed_forward: int = 256
    dropout: float = 0.2
    window: int = 7
    lr: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 200
    patience: int = 20
    batch_size: int = 8
    grad_clip: float = 1.0
    workers: int = 0
    lambda_tasks: tuple = (1.0, 0.7, 0.7, 0.5, 0.5)
    risk_threshold: float = 0.50
    resilience_threshold: float = 0.45
    ahp_weights: tuple = (0.424, 0.227, 0.227, 0.122)
    propagation_gamma: tuple = (0.5, 0.3, 0.2)
    max_propagation_hops: int = 2
    dep_threshold: float = 0.05
    loss_huber_delta: float = 1.0
    base_seed: int = 11
    device: str = "auto"
    report_dpi: int = 600
    demo_days: int = 36
    demo_nodes_per_type: int = 8

    @classmethod
    def from_yaml(cls, path: str | Path | None = None, overrides: dict | None = None):
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
        data = data or {}
        if "config" in data:
            data = data["config"]
        data.update(overrides or {})
        permitted = cls.__dataclass_fields__
        unknown = set(data) - set(permitted)
        if unknown:
            raise ValueError(f"Unrecognised configuration keys: {sorted(unknown)}")
        settings = cls(**data)
        settings.validate()
        return settings

    def validate(self):
        if self.hidden % self.graph_heads or self.hidden % self.temporal_heads:
            raise ValueError("hidden size must be divisible by attention head counts")
        if self.window < 2 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("window>=2, epochs>=1, batch_size>=1 are required")
        if not 0 <= self.risk_threshold <= 1 or not 0 <= self.resilience_threshold <= 1:
            raise ValueError("thresholds must be in [0,1]")
        if abs(sum(self.ahp_weights)-1) > 1e-6:
            raise ValueError("AHP weights must sum to 1")

    def json(self) -> dict:
        return asdict(self)

    def save(self, path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(self.json(),indent=2),encoding="utf-8")
