# Two-Month Lesson Plan

## Foundation Track: Neural Networks and Spiking Neural Networks

This track runs alongside Weeks 1-4. It gives you the substrate for the transformer, scaling-law, and kernel work.

- Neural networks: perceptrons, MLPs, activations, losses, backpropagation, initialization, optimizers, regularization, and train/eval splits.
- JAX implementation practice: write a tiny MLP training loop from scratch before relying on Flax abstractions.
- Spiking neural networks: leaky integrate-and-fire dynamics, membrane potential, thresholds, reset behavior, spike trains, surrogate gradients, and event-driven computation.
- Comparison memo: explain when dense neural nets are the right abstraction, when SNNs are interesting, and why most frontier-lab LLM work still uses dense differentiable models.
- Mini deliverable: run a LIF neuron simulation and write one page connecting SNNs to neuromorphic hardware, latency, sparsity, and energy efficiency.

## Week 1: Orient and Set Up

- Set up JAX, Flax, Optax, and the repo.
- Finish a basic JAX tutorial.
- Complete neural-network foundations: perceptron, MLP, loss functions, and backprop.
- Draft the target role memo.
- Start the target team and outreach tracker.

## Week 2: Build the 10M Transformer

- Generate synthetic addition data.
- Implement a tiny MLP in JAX on a toy task to make backprop concrete.
- Implement tokenizer and decoder-only transformer.
- Train a baseline model.
- Track loss and accuracy.

## Week 3: Scaling Laws

- Run sweeps over model size, data size, and training steps.
- Derive and test Chinchilla-style claims for the toy task.
- Study how optimization behavior changes with width, depth, learning rate, and initialization.
- Write a scaling-law report draft.

## Week 4: Dense vs MoE

- Study dense vs MoE compute assumptions.
- Add a short comparison of dense neural nets, sparse MoE routing, and spiking sparsity.
- Implement or sketch a small MoE variant.
- Identify a fused operation worth benchmarking.

## Week 5: Pallas / Kernel Work

- Learn enough Pallas to write a small kernel.
- Benchmark a fused operation against a baseline.
- Explain the bottleneck model.

## Week 6: Research Context

- Read and write notes on FlashAttention, quantization, kernels/DSLs, and KV-cache work.
- Write a synthesis memo about recurring performance-work patterns.

## Week 7: Agent Work Backup Track

- Build one controlled agent experiment.
- Define task, dataset, metrics, ablations, and failure taxonomy.

## Week 8: Package and Apply

- Polish README and technical memo.
- Prepare resume bullets.
- Send 10 targeted outreach/application attempts.
