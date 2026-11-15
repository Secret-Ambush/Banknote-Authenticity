# Coursework requirements and Phase 1 boundary

## Mandatory later coursework work

- Implement a configurable multilayer feedforward ANN from scratch.
- Implement PSO from scratch, using Algorithm 39 (Section 3.5) of *Essentials of Metaheuristics* and informants.
- Encode an ANN's weights and biases in each particle, optimise binary-classification parameters with PSO, and investigate architectures and PSO hyperparameters through repeated experiments.
- Use no TensorFlow, Keras, PyTorch, pre-built ANN implementations, or pre-built PSO implementations.
- Include Sigmoid, ReLU, and Tanh activation functions. Backpropagation is not required.

## Phase 1 mandatory work

- Download the original UCI Banknote Authentication dataset unchanged.
- Validate, inspect, class-wise stratify, split, standardise, save, document, and test the dataset pipeline.
- Ensure standardisation learns statistics only from training data.

## Optional extensions

Visualisations, additional data diagnostics, alternative split ratios, and richer command-line interfaces are optional only if they do not compromise the required pipeline. None have been added in Phase 1.

## Explicitly out of scope for Phase 1

No ANN or PSO code, optimisation, experimental runs, or performance metrics are included. In particular, there are no neurons, weight matrices, forward propagation, activation functions, backpropagation, particle classes, velocity updates, informant selection, fitness functions, or ANN/PSO integration.
