# F21BC Biologically Inspired Computation

This MSc Artificial Intelligence coursework project will implement a configurable multilayer feedforward ANN and an informant-based PSO optimiser from scratch for Banknote Authentication classification. Phase 1 is deliberately limited to reproducible dataset preparation; ANN, PSO, experiments, and performance results are not implemented.

## Coursework requirements

The later coursework stages require a from-scratch ANN, a from-scratch informant-based PSO following Algorithm 39 in *Essentials of Metaheuristics*, and PSO-based optimisation of the complete ANN parameter vector. Framework ANN/PSO implementations are not permitted. See [the detailed requirements](docs/coursework_requirements.md).

## Structure

```
data/raw/              Original UCI file (downloaded locally)
data/processed/        Prepared compressed NPZ file (generated locally)
src/data/              Download, inspection, and preprocessing code
src/ann/               Phase 2 NumPy-only ANN forward-propagation code
src/pso/               Empty placeholder for a later phase
tests/                 Phase 1 pipeline tests
experiments/results/   Placeholder for future work
docs/                  Coursework notes
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset and preparation

The original headerless dataset is downloaded from the [UCI Banknote Authentication repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication) and saved unchanged to `data/raw/data_banknote_authentication.txt`.

```bash
python -m src.data.download
python -m src.data.inspection
python -m src.data.preprocessing
pytest -q
```

If download fails, manually obtain `data_banknote_authentication.txt` from the UCI source and place it in `data/raw/`, then run the remaining commands.

Preprocessing uses a seeded (`42` by default), class-wise stratified 70:30 split. Feature means and standard deviations are computed exclusively from training data, then applied to both partitions. The generated `data/processed/banknote_data.npz` contains `X_train`, `X_test`, `y_train`, `y_test`, `mean`, `std`, `random_seed`, and `test_fraction`.

Load it in Python with:

```python
from src.data.preprocessing import load_prepared_dataset
data = load_prepared_dataset()
```

## Current status

Phase 1 data preparation and Phase 2 ANN forward propagation are implemented. PSO, ANN training, optimisation, and experiments are not implemented.

## ANN forward propagation (Phase 2)

The NumPy-only `NeuralNetwork` supports configurable fully connected architectures, `sigmoid`, `relu`, and `tanh` hidden activations, and sigmoid output probabilities. It performs forward propagation and threshold-based prediction only; it does not train or update parameters.

```python
from src.ann.neural_network import NeuralNetwork
from src.ann.parameter_manager import get_parameter_vector, set_parameter_vector

network = NeuralNetwork([4, 8, 1], hidden_activations="relu", random_seed=42)
probabilities = network.forward(X_train)
predictions = network.predict(X_train)

# This vector format is ready for a later PSO phase.
parameters = get_parameter_vector(network)
set_parameter_vector(network, parameters)
```

The predefined architecture sizes are available in `src.ann.architecture.ARCHITECTURES`. Pass a single hidden activation name to use it for every hidden layer, or a sequence such as `("relu", "tanh")` for a multi-hidden-layer network. Parameter vectors store each layer's weights followed by its biases, from input to output.
