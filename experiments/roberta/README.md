# Monarch Attention for RoBERTa

Testing Monarch Attention on extractive question answering using [RoBERTa-base fine-tuned on SQuAD](https://huggingface.co/csarron/roberta-base-squad-v1). No training is required — the pretrained fine-tuned checkpoint is used directly.

## Benchmarking

Run benchmarks across attention mechanisms:

```bash
python -m experiments.roberta.benchmark
```

Results are saved as JSON files in `experiments/roberta/results/`. The script evaluates:

- **Softmax** (baseline)
- **Monarch Attention** on layers {0,1,2,3,8,9,10,11} with block_size in {24, 48, 96, 128}
- **Performer** with rank in {32, 64, 96, 128, 160, 192}
- **Nystromformer** with rank in {16, 32, 48, 64}
- **Cosformer**
- **LinearAttention**

Evaluation uses 1024 samples with batch size 8.

## Plotting

Generate the F1 vs. FLOPs figure (combined with ViT results):

```bash
python -m experiments.figures.plot_vit_roberta
```

Output: `experiments/figures/vit_roberta_results.pdf`

This produces a two-panel figure with ViT top-5 accuracy (left) and RoBERTa F1 (right).

## Attention Visualization

Generate attention map comparisons across mechanisms:

```bash
python -m experiments.roberta.visualize_attentions
```

Output: `experiments/figures/roberta_attention_maps.pdf`

The script extracts query/key tensors (cached to `experiments/roberta/query.pt`, `experiments/roberta/key.pt`, `experiments/roberta/attention_mask.pt`) and visualizes attention maps at layer 5, head 5 for Softmax, Monarch Attention, Performer, Nystromformer, Cosformer, and LinearAttention.
