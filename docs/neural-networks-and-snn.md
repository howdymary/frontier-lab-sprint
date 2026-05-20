# Neural Networks and Spiking Neural Networks Track

This is the foundation layer for the frontier-lab sprint. It should make the transformer work feel less like memorizing APIs and more like understanding the machinery.

## Module 1: Classical Neural Networks

Goal: understand how differentiable models learn.

Topics:

- Perceptrons and linear classifiers.
- MLPs and universal approximation intuition.
- Activations: sigmoid, tanh, ReLU, GELU.
- Losses: cross entropy and mean squared error.
- Backpropagation and the chain rule.
- Initialization, normalization, gradient flow, and optimizer behavior.
- SGD, momentum, Adam, learning-rate schedules.
- Generalization, regularization, train/eval splits, and overfitting.

Exercises:

- Derive gradients for a two-layer MLP by hand.
- Implement a tiny MLP in JAX without Flax.
- Train it on XOR or a synthetic classification task.
- Write a short note explaining what changed when learning rate, width, depth, and activation changed.

Deliverable:

- One notebook or script with a from-scratch JAX MLP.
- One short memo: "What backprop is actually doing."

## Module 2: Bridge to Transformers

Goal: connect neural-network basics to the transformer project.

Topics:

- Embeddings as learned lookup tables.
- Sequence modeling and autoregressive loss.
- Attention as content-based routing.
- Residual streams, layer norm, MLP blocks, and logits.
- Why transformers are dense differentiable programs.

Exercises:

- Write shapes for every tensor in a tiny decoder-only transformer.
- Implement one attention block in JAX.
- Explain where compute and memory are spent in a forward pass.

Deliverable:

- Shape walkthrough for the addition transformer.
- Attention block implementation or annotated pseudocode.

## Module 3: Spiking Neural Networks

Goal: understand SNNs as a different computational abstraction, not a replacement buzzword for transformers.

Topics:

- Leaky integrate-and-fire neuron dynamics.
- Membrane potential, input current, threshold, spike, reset, and refractory period.
- Spike trains as temporal/event-based representations.
- Rate coding vs temporal coding.
- Surrogate gradients for training through non-differentiable spikes.
- Backpropagation through time for recurrent spiking systems.
- Neuromorphic hardware motivation: sparsity, latency, and energy efficiency.
- Practical limitation: most current frontier LLM work uses dense differentiable accelerators, so SNNs are mainly a conceptual and hardware-adjacent extension for this sprint.

Exercises:

- Simulate one leaky integrate-and-fire neuron under constant and pulsed input.
- Plot membrane potential and spikes.
- Explain how thresholding creates a non-differentiability.
- Sketch how surrogate gradients make training possible.
- Compare dense neural-network activation sparsity, MoE routing sparsity, and SNN spike sparsity.

Deliverable:

- One runnable LIF simulation.
- One page: "What SNNs clarify about sparsity, timing, and hardware."

## Integration With The Two-Month Sprint

Do not let this become a separate rabbit hole. The SNN track is useful because it sharpens your model of neural computation, sparsity, and hardware constraints. The frontier-lab hiring artifact should still stay centered on the JAX transformer, scaling-law analysis, and performance benchmark.

Suggested pacing:

- Week 1: neural-network basics and backprop.
- Week 2: JAX MLP and transformer shape walkthrough.
- Week 3: optimization behavior and gradient flow.
- Week 4: SNN concepts and LIF simulation.
- Week 5 onward: fold sparsity/hardware insights into MoE, kernel, and performance notes.

