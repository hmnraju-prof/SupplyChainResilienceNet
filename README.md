# SupplyChainResilienceNet

### A Graph-Based Framework for Supply Chain Disruption Prediction and Adaptive Recovery Optimisation

SupplyChainResilienceNet is an integrated graph-based framework designed to predict supply chain disruptions, analyse their potential impact, estimate operational risks, and recommend feasible recovery actions.

The framework combines **Heterogeneous Graph Attention Networks (GAT), Temporal Transformers, Multi-Task Learning, AHP-Based Resilience Scoring, and Mixed-Integer Linear Programming (MILP)** to support intelligent and adaptive supply chain management.

It models complex relationships between suppliers, factories, warehouses, ports, logistics hubs, and demand regions while incorporating historical operational information and time-varying dependencies.

---

# 1. Project Overview

Supply chains consist of interconnected entities whose operational performance depends on multiple relationships and external factors. Disruptions affecting one entity can propagate through connected entities, resulting in delays, operational losses, and reduced supply chain resilience.

SupplyChainResilienceNet addresses these challenges through a unified graph-temporal learning and optimisation framework.

The proposed framework performs five major functions:

1. **Disruption Prediction:** Estimates the probability of supply chain disruptions.
2. **Disruption Classification:** Identifies disruption categories and severity levels.
3. **Impact Estimation:** Predicts operational delays and financial losses.
4. **Resilience Assessment:** Evaluates supply chain resilience using an Analytical Hierarchy Process (AHP).
5. **Recovery Optimisation:** Identifies feasible mitigation actions using constraint-aware optimisation.

---

# 2. Key Features

- Six heterogeneous supply chain node types.
- Six logical relationship types representing supply chain interactions.
- Dynamic edge attributes for dependency, delay, historical disruption, and capacity.
- Relation-specific, edge-aware Graph Attention Network.
- Temporal Transformer for historical dependency modelling.
- Multi-task learning with five prediction objectives.
- Binary disruption probability estimation.
- Nine-category disruption classification.
- Four-level disruption severity classification.
- Operational delay prediction in days.
- Financial loss estimation in thousand USD.
- AHP-based supply chain resilience scoring.
- Five candidate recovery and mitigation strategies.
- Joint Mixed-Integer Linear Programming (MILP) optimisation.
- Shared-resource and budget constraint management.
- Logistic Regression and Random Forest baselines.
- Graph-temporal GRU baseline.
- Rolling temporal evaluation and cross-validation.
- Architectural ablation and threshold sensitivity analysis.
- Performance evaluation, statistical testing, and visualisation.
- Model checkpoint saving and reloading.
- Automated unit and integration testing.
- Google Colab support.

---

# 3. Proposed Architecture

The SupplyChainResilienceNet framework consists of interconnected processing stages.

## 3.1 Heterogeneous Supply Chain Graph Construction

The supply chain is represented as a heterogeneous temporal graph.

### Node Types

The framework considers six types of supply chain entities:

| Node Type | Description |
|---|---|
| Supplier | Provides raw materials and components |
| Factory | Performs manufacturing and production |
| Warehouse | Stores and manages inventory |
| Port | Supports maritime transportation |
| Logistics Hub | Coordinates transportation and distribution |
| Demand Region | Represents downstream demand locations |

### Relationship Types

Six logical relationship types represent interactions between supply chain entities.

| Relationship | Description |
|---|---|
| Procurement | Supplier procurement relationships |
| Shipment | Transportation and shipment relationships |
| Production | Manufacturing and production dependencies |
| Inventory | Inventory-related dependencies |
| Recovery | Recovery-related relationships |
| Propagation | Disruption propagation relationships |

Each relationship can incorporate time-varying attributes such as dependency strength, transportation delay, historical disruption information, and operational capacity.

## 3.2 Relation-Specific Graph Attention Network

The graph representation is processed using a relation-specific, edge-aware Graph Attention Network.

The network captures interactions between heterogeneous supply chain entities and learns the importance of connected nodes and their relationships.

The architecture consists of:

- Three graph attention layers.
- Four attention heads.
- 128 hidden dimensions.
- Relation-specific message aggregation.
- Edge-aware attention mechanisms.

The graph attention component learns spatial and relational dependencies within the supply chain network.

## 3.3 Temporal Transformer

The temporal Transformer captures historical operational dependencies and changes in supply chain behaviour.

It processes historical graph representations to identify temporal patterns associated with disruption risks.

Configuration:

- Two Transformer encoder layers.
- Four temporal attention heads.
- 256 feed-forward dimensions.
- Seven-day historical observation window.
- One-day-ahead prediction.

The temporal component complements graph learning by capturing changes in supply chain conditions over time.

## 3.4 Multi-Task Prediction Framework

The framework uses multiple prediction heads to perform five related tasks.

| Task | Prediction Objective |
|---|---|
| Task 1 | Binary disruption prediction |
| Task 2 | Disruption type classification |
| Task 3 | Disruption severity classification |
| Task 4 | Operational delay prediction |
| Task 5 | Financial loss prediction |

The shared graph-temporal representation supports multiple prediction objectives within a unified learning framework.

---

# 4. Resilience Scoring

SupplyChainResilienceNet incorporates an Analytical Hierarchy Process (AHP) to calculate a supply chain resilience score.

The resilience assessment considers four factors:

- Disruption probability.
- Disruption severity.
- Expected operational delay.
- Expected financial loss exposure.

The resilience score is calculated as:

**RS = 1 − (0.424P + 0.227S + 0.227D + 0.122C)**

Where:

- RS = Resilience Score
- P = Disruption Probability
- S = Disruption Severity
- D = Normalised Operational Delay
- C = Expected Loss Exposure

The AHP weights are:

| Parameter | Weight |
|---|---:|
| Disruption Probability | 0.424 |
| Disruption Severity | 0.227 |
| Operational Delay | 0.227 |
| Expected Loss Exposure | 0.122 |

The scoring mechanism provides an action-independent assessment of supply chain resilience.

---

# 5. Adaptive Recovery Optimisation

The framework incorporates a Mixed-Integer Linear Programming (MILP) optimisation component using SciPy and HiGHS.

It evaluates five candidate recovery strategies:

1. Rerouting.
2. Supplier Substitution.
3. Production Reallocation.
4. Buffer Stock Allocation.
5. Recovery Activation.

The optimisation component evaluates candidate actions based on:

- Resource availability.
- Recovery lead time.
- Service requirements.
- Shared-resource capacities.
- Operational feasibility.
- Budget constraints.
- Estimated mitigation effects.
- Direct recovery costs.

A no-action alternative is also available.

The optimisation process selects feasible recovery decisions while satisfying the specified operational constraints.

The recovery component also supports operational state updates to analyse the effects of selected actions.

---

# 6. Dataset Sources

The framework supports three public supply chain and logistics data sources.

| Dataset | Source | Application |
|---|---|---|
| SupplyGraph | [GitHub Repository](https://github.com/ciol-researchlab/SupplyGraph) | Product and operational time-series information |
| Supply Chain Disruption and Recovery | [Kaggle Dataset](https://www.kaggle.com/datasets/likithagedipudi/supply-chain-disruption-and-recovery-dataset) | Disruption categories, severity, delays, and loss information |
| Maritime Port Performance | [Kaggle Dataset](https://www.kaggle.com/datasets/jeleeladekunlefijabi/maritime-port-performance-dataset) | Maritime operational indicators |

Dataset files must be obtained separately from their respective sources.

## 6.1 Dataset Organisation

Place the extracted CSV files in the following directories:

```text
data/
└── raw/
    ├── supplygraph/
    ├── disruptions/
    └── maritime/
```

## 6.2 Data Preparation

The framework provides an automated data preparation pipeline.

```bash
python scripts/download_data.py --dataset all --directory data/raw
```

Audit the downloaded datasets:

```bash
python run_all.py audit --raw-root data/raw --out outputs/raw_audit.json
```

Convert public datasets into the canonical graph representation:

```bash
python run_all.py convert-public --raw-root data/raw --out data/canonical --seed 11
```

Prepare the processed dataset:

```bash
python run_all.py prepare --canonical data/canonical --out data/processed
```

The public-data adapter supports cross-source integration. Where verified identifiers are unavailable, illustrative entity relationships are constructed and should be replaced with verified linkage information for real operational deployment.

---

# 7. Canonical Dataset Structure

The framework also supports user-provided, verified supply chain relationships through five canonical CSV files.

| File | Required Columns |
|---|---|
| entities.csv | entity_id, entity_type |
| relations.csv | source_id, target_id, relation_type |
| operations.csv | day, entity_id |
| events.csv | entity_id, start_date |
| ports.csv | entity_id, and day or period |

Optional columns provide additional operational information, including:

- Region and industry.
- Supplier tier and capacity.
- Dependency strength.
- Production and inventory.
- Delivery and lead time.
- Disruption type and severity.
- Delay and financial loss.
- Port turnaround and vessel calls.

The canonical loader validates entity identifiers, graph relationships, dates, feature dimensions, and prediction labels.

---

# 8. Real-Data Port Risk Prediction

The framework includes a separate port-turnaround prediction task using observed daily port-level records.

Required CSV columns:

```text
day
port_id
turnaround_days
```

Optional columns:

```text
vessel_calls
cargo_capacity
port_efficiency
```

Minimum data requirements:

- At least 20 fully observed daily dates.
- At least two actual ports.
- Valid port identifiers.
- Daily turnaround observations.

The model estimates next-day port turnaround exceedance using a training-only, per-port quantile.

Run the port prediction task:

```bash
python run_all.py port-task --csv /path/to/real_daily_port_records.csv --config configs/paper.yaml --out outputs/real_port
```

The port task uses actual daily port records and does not treat country-level or semiannual observations as daily port measurements.

---

# 9. Model Configuration

The main configuration is maintained in:

```text
configs/paper.yaml
```

The default configuration includes:

| Parameter | Configuration |
|---|---:|
| Graph Attention Layers | 3 |
| Graph Attention Heads | 4 |
| Hidden Dimensions | 128 |
| Transformer Layers | 2 |
| Temporal Attention Heads | 4 |
| Feed-Forward Dimensions | 256 |
| Historical Window | 7 Days |
| Optimiser | AdamW |
| Learning Rate | 0.001 |
| Maximum Epochs | 200 |
| Early Stopping Patience | 20 |
| Gradient Clipping | 1.0 |

## 9.1 Loss Functions

The multi-task learning framework incorporates:

- Weighted Binary Cross-Entropy for disruption prediction.
- Weighted Categorical Cross-Entropy for disruption type classification.
- Weighted Categorical Cross-Entropy for severity classification.
- Huber Regression for operational delay estimation.
- Huber Regression for financial loss estimation.

Target scaling and masking are applied according to the availability of the corresponding observations.

---

# 10. Baseline Models and Comparative Evaluation

The framework includes the following baseline implementations:

| Model | Description |
|---|---|
| Logistic Regression | Traditional classification baseline |
| Random Forest | Tree-based prediction baseline |
| Mean Aggregation + GRU | Graph-temporal baseline |
| SupplyChainResilienceNet | Relation-specific Graph Attention + Temporal Transformer |

The graph-temporal baseline uses mean graph aggregation and a GRU instead of the proposed graph attention and Transformer combination.

## 10.1 Evaluation Methods

The framework supports:

- Rolling temporal train-validation-test splits.
- Five-seed experiments.
- Five-fold rolling-origin evaluation.
- Binary classification metrics.
- Macro-F1 classification metrics.
- Regression evaluation.
- Architectural ablation studies.
- Threshold sensitivity analysis.
- Wilcoxon signed-rank testing.
- Holm multiple-comparison correction.
- Bootstrap confidence intervals.
- Rank-biserial effect sizes.

## 10.2 Ablation Studies

Seven architectural and feature ablations are implemented:

1. Without maritime features.
2. Without historical disruption information.
3. Without propagation relationships.
4. Mean graph aggregation.
5. Without graph aggregation.
6. Without temporal encoder.
7. GRU instead of Transformer.

---

# 11. Installation and Requirements

### Prerequisites

- Python 3.10–3.13.
- PyTorch.
- NumPy.
- Pandas.
- SciPy.
- Scikit-learn.
- Matplotlib.
- PyYAML.
- Pytest.

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SupplyChainResilienceNet.git
```

Navigate to the project directory:

```bash
cd SupplyChainResilienceNet
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 12. Running the Framework

## 12.1 Execute the Demonstration Pipeline

```bash
python run_all.py demo --config configs/demo.yaml --out outputs/demo --baselines --sensitivity
```

## 12.2 Train the Main Model

After preparing the canonical dataset:

```bash
python run_all.py train --input data/processed --config configs/paper.yaml --out outputs/paper --baselines
```

## 12.3 Run Cross-Validation

```bash
python run_all.py crossval --input data/processed --config configs/paper.yaml --folds 5 --seeds 11 23 37 51 73 --compare-stgcn --out outputs/crossval
```

## 12.4 Run Ablation and Sensitivity Experiments

```bash
python run_all.py train --input data/processed --config configs/paper.yaml --baselines --ablate --sensitivity --out outputs/full_study
```

## 12.5 Execute Automated Tests

```bash
python -m pytest -q
```

---

# 13. Output Files

The framework organises its outputs into structured directories.

```text
outputs/
├── demo/
├── paper/
├── crossval/
├── full_study/
└── real_port/
```

Typical outputs include:

| Output | Description |
|---|---|
| configuration.json | Model configuration |
| split_indices.json | Temporal data split information |
| data_manifest.json | Dataset metadata |
| results.json | Evaluation results |
| predictions.csv | Model predictions |
| recovery_actions.csv | Selected recovery decisions |
| simulated_state_changes.csv | Operational state updates |
| training_history.csv | Training and validation history |
| model.pt | Saved model checkpoint |
| training_curves.png | Training performance visualisation |
| disruption_metrics.png | Disruption prediction metrics |
| recovery_simulation.png | Recovery analysis visualisation |
| threshold_sensitivity.csv | Threshold sensitivity results |

---

# 14. Repository Structure

```text
SupplyChainResilienceNet/
│
├── configs/
│   ├── demo.yaml
│   └── paper.yaml
│
├── data/
│   ├── raw/
│   │   ├── supplygraph/
│   │   ├── disruptions/
│   │   └── maritime/
│   ├── canonical/
│   ├── processed/
│   └── splits/
│
├── outputs/
│
├── notebooks/
│   └── Colab_Quick_Start.ipynb
│
├── scripts/
│   └── download_data.py
│
├── scrn/
│   ├── __init__.py
│   ├── config.py
│   ├── schema.py
│   ├── synthetic.py
│   ├── data.py
│   ├── model.py
│   ├── losses.py
│   ├── training.py
│   ├── scoring.py
│   ├── optimization.py
│   ├── simulation.py
│   ├── baselines.py
│   ├── experiments.py
│   ├── port_task.py
│   ├── metrics.py
│   ├── plots.py
│   ├── statistics.py
│   └── cli.py
│
├── tests/
│   ├── test_data.py
│   ├── test_graph.py
│   ├── test_port.py
│   ├── test_scoring_optimization.py
│   ├── test_statistics.py
│   └── test_training.py
│
├── run_all.py
└── requirements.txt
```

---

# 15. Google Colab Support

The project includes a Google Colab starter notebook.

```text
notebooks/Colab_Quick_Start.ipynb
```

To execute the framework in Google Colab:

1. Open the starter notebook.
2. Upload the project ZIP.
3. Extract the repository.
4. Install the required dependencies.
5. Execute the demonstration pipeline.
6. Review the generated evaluation outputs.

For GPU execution, install the appropriate PyTorch version for the available CUDA environment.

PyTorch installation instructions:

https://pytorch.org/get-started/locally/

---

# 16. Technologies and Libraries

The implementation uses the following technologies:

| Technology | Application |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning implementation |
| NumPy | Numerical computation |
| Pandas | Dataset processing |
| SciPy | MILP optimisation and statistical analysis |
| HiGHS | Mathematical optimisation solver |
| Scikit-learn | Baseline models and evaluation |
| Matplotlib | Data and performance visualisation |
| PyYAML | Model configuration |
| Pytest | Automated testing |
| Google Colab | Cloud-based execution |
| PyTorch Geometric | Optional heterogeneous graph export |


---


# 18. Acknowledgements

The framework builds upon established concepts in:

- Heterogeneous Graph Neural Networks.
- Graph Attention Networks.
- Transformer-based Temporal Learning.
- Multi-Task Learning.
- Analytical Hierarchy Process.
- Mixed-Integer Linear Programming.
- Supply Chain Risk and Resilience Modelling.

---

