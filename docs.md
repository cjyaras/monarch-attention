## How it works
Sparsemax attention can be computed as:
$$\mathrm{sparsemax}(QK^\top) = \argmin_{A \in \Delta} f(A):= \frac{1}{2}\|A-QK^\top\|_F^2.$$
We can evaluate 
$$f(A) = \frac{1}{2}\mathrm{tr}(A^\top A) - \mathrm{tr}(Q^\top(AK)) + c$$
meaning that for any structured matrix $A$ that admits *subquadratic*

(1) storage
(2) matmul
(3) trace of gram matrix
(4) "projection" onto simplex

we can compute gradients of the sparsemax objective efficiently and (hopefully) find a structure-optimal matrix in a few iterations of gradient descent. 

This gives us a flexible framework for finding high-quality structured attention approximations that can transfer with no additional training to existing pre-trained bi-directional transformers.

##### Applications:
(1) Vision transformers
- seemingly little to no loss in accuracy on ImageNet (196 length sequences) with \~40\% reduction in FLOPs with Monarch attention
- **look at long-range arena**

(2) BERT-style language modeling
- have an example script with sparsemax roberta for mask-filling (256 length sequence) with \~20\% reduction in FLOPs (doesn't work on some examples e.g. question answering, still debugging). **certain layers might be difficult to approximate with structured attention?**

- will probably work better on GLUE classification tasks

(3) DNA sequence models
- haven't started yet


##### Baselines:
Little to no works for train-free, transferable structured attention.

(1) Kernel attention (e.g. performer https://arxiv.org/pdf/2009.14794)
(2) Low-rank attention via JL-lemma (e.g. linformer https://arxiv.org/pdf/2006.04768)
(3) Learnable attention (e.g. hedgehog https://arxiv.org/pdf/2402.04347 or lolcats https://arxiv.org/pdf/2410.10254) - __not a completely fair comparison since the attention mechansism is distilled from downstream task.__