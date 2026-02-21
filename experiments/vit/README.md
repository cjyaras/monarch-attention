# Monarch Attention for ViT

Testing Monarch Attention on image classification using [ViT-base](https://huggingface.co/google/vit-base-patch16-224) on ImageNet-1k. No training is required — the pretrained model is used directly.

## Benchmarking

Run benchmarks across attention mechanisms:

```bash
python -m experiments.vit.benchmark
```

Results are saved as JSON files in `experiments/vit/results/`. The script evaluates:

- **Softmax** (baseline)
- **Monarch Attention** with block_size 14 and num_steps in {1, 2, 3}
- **Performer** with rank in {16, 32, 48, 64, 80, 96}
- **Nystromformer** with rank in {16, 24, 32, 40}
- **Cosformer**
- **LinearAttention**

Evaluation uses 1024 samples with batch size 8, measuring top-5 accuracy.

## Plotting

Generate the accuracy vs. FLOPs figure (combined with RoBERTa results):

```bash
python -m figures.plot_vit_roberta
```

Output: `figures/vit_roberta_results.pdf`

This produces a two-panel figure with ViT top-5 accuracy (left) and RoBERTa F1 (right).

## Attention Visualization

Generate attention map comparisons across mechanisms:

```bash
python -m experiments.vit.visualize_attentions
```

Output: `figures/attention_maps.pdf`

The script extracts query/key tensors (cached to `experiments/vit/query.pt`, `experiments/vit/key.pt`) and visualizes attention maps at layer 3, head 3 for Softmax, Monarch Attention (block_size=14, num_steps=2), Performer, Nystromformer, Cosformer, and LinearAttention.
