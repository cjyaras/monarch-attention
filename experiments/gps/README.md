# Monarch Attention for GPS

Testing Monarch Attention on graph classification using a [GPS](https://arxiv.org/abs/2205.12454) (General, Powerful, Scalable) model on the Actor dataset from PyTorch Geometric.

## 1. Training

Train the baseline GPS model:

```bash
python -m experiments.gps.train
```

This saves the trained model to `experiments/gps/gps_model.pth`. Training uses AdamW with a learning rate of 5e-4 for 1500 steps.

## 2. Benchmarking

Run benchmarks across attention mechanisms:

```bash
python -m experiments.gps.benchmark
```

Results are saved as JSON files in `experiments/gps/results/`. The script evaluates:

- **Softmax** (baseline)
- **Monarch Attention** with block size 128 and num_steps in {1, 2, 3, 4}
- **Nystromformer** with rank in {128, 192, 256, 320}

## 3. Plotting

Generate the accuracy vs. FLOPs figure:

```bash
python -m experiments.figures.plot_gps
```

Output: `experiments/figures/gps_results.pdf`
