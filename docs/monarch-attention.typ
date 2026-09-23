#set page(margin: 0.5in)
#set text(size: 11pt)
#set math.equation(numbering: "(1)")
#let ma = raw("MonarchAttention")

#align(center)[
  #text(size: 16pt, weight: "bold")[#ma: Sub-Quadratic Attention via Monarch Matrices]
]

= Setup

For queries, keys, and values $bold(Q), bold(K), bold(V) in RR^(N times d)$, one head of softmax attention computes
$ bold(O) = "softmax"(bold(Q) bold(K)^top) bold(V), $
with softmax applied across rows. Forming the $N times N$ attention matrix costs $Theta(N^2 d)$. The goal of #ma is to find a structured matrix $bold(M) approx "softmax"(bold(Q) bold(K)^top)$ in $o(N^2 d)$ time, without ever materializing the full attention matrix, and then compute $bold(O) approx bold(M) bold(V)$ cheaply.

Throughout, a summation without written indices runs over all indices in its summand.

= Preliminaries

*Variational form of softmax.* For $bold(z) in RR^N$ and temperature $c > 0$,
$ "softmax"(bold(z) \/ c) = arg max_(bold(p) in Delta^N) sum_n bold(p)_n bold(z)_n - c sum_n bold(p)_n log bold(p)_n, $ <eq:var>
where $Delta^N$ is the probability simplex. Softmax is the unique maximizer of a linear term plus scaled entropy.

*Monarch matrices.* Let $N = m b$. A matrix $bold(B) in RR^(N times N)$ is _block rank-one_ if there are $bold(L) in RR^(b times m times m)$ and $bold(R) in RR^(m times b times b)$ with
$ bold(B)_((j l)(k i)) = bold(L)_(j k l) bold(R)_(k j i), quad i, j in [b], quad k, l in [m], $
where $(j l) = (j-1) m + l$ and $(k i) = (k-1) b + i$. That is, $bold(B)$ is a $b times m$ grid of $m times b$ blocks, and block $(j, k)$ is the outer product $bold(L)_(j k :) bold(R)_(k j :)^top$. A _Monarch_ matrix swaps the two components of the row index:
$ bold(M)_((l j)(k i)) = bold(L)_(j k l) bold(R)_(k j i), quad (l j) = (l-1) b + j. $ <eq:monarch>
Two properties matter:
- *Storage.* $bold(L)$ and $bold(R)$ hold $N(m + b)$ entries in total, which is $2 N sqrt(N)$ for $m = b = sqrt(N)$.
- *Multiplication.* For $bold(V) in RR^(N times d)$, $bold(M) bold(V)$ is two batches of small matmuls,
  $ bold(Y)_(j k v) = sum_i bold(R)_(k j i) bold(V)_((k i) v), quad (bold(M) bold(V))_((l j) v) = sum_k bold(L)_(j k l) bold(Y)_(j k v), $ <eq:matmul>
  costing $Theta(N (m + b) d) = Theta(N sqrt(N) d)$ when $m = b = sqrt(N)$.

An exact projection of the attention matrix onto Monarch matrices would cost $O(N^2 sqrt(N))$ and require the full attention matrix. #ma avoids both by optimizing the variational objective directly over the factors.

#block(breakable: false)[
= Derivation

By @eq:var applied row-wise (with $c = 1$), softmax attention is the solution of
$ "softmax"(bold(Q) bold(K)^top) = arg max_(bold(P) in RR^(N times N)) quad &sum bold(P)_(m n) bold(Q)_(m v) bold(K)_(n v) - sum bold(P)_(m n) log bold(P)_(m n) \
  "s.t." quad &bold(P)_(m n) >= 0, quad sum_n bold(P)_(m n) = 1. $ <eq:attn-var>
]
We now constrain $bold(P)$ to be Monarch, $bold(P)_((l j)(k i)) = bold(L)_(j k l) bold(R)_(k j i)$ with $bold(L) in RR^(b times m times m)$ and $bold(R) in RR^(m times b times b)$, which gives
$ max_(bold(L), bold(R)) quad &sum bold(L)_(j k l) bold(R)_(k j i) bold(Q)_((l j) v) bold(K)_((k i) v) - sum bold(L)_(j k l) bold(R)_(k j i) log (bold(L)_(j k l) bold(R)_(k j i)) \
  "s.t." quad &bold(L)_(j k l) bold(R)_(k j i) >= 0, quad sum_(k, i) bold(L)_(j k l) bold(R)_(k j i) = 1. $ <eq:monarch-var>
The goal is to rewrite this objective so that maximizing over $bold(L)$ (or $bold(R)$) alone looks exactly like @eq:var, giving closed-form alternating updates. This takes two steps: separate the constraints, then separate the entropy.

== Separable constraints

We replace the constraints in @eq:monarch-var with the equivalent _separable_ form
$ {bold(L)_(j k l) >= 0, quad sum_k bold(L)_(j k l) = 1} quad "and" quad {bold(R)_(k j i) >= 0, quad sum_i bold(R)_(k j i) = 1}. $ <eq:sep>
This is without loss of generality: any $(bold(L), bold(R))$ satisfying the original constraints can be renormalized to satisfy @eq:sep without changing the products $bold(L)_(j k l) bold(R)_(k j i)$, so the two constraint sets parametrize exactly the same Monarch matrices and the global optimum is preserved.

== Separable entropy

Under @eq:sep the entropy term also separates. Splitting $log(bold(L)_(j k l) bold(R)_(k j i)) = log bold(L)_(j k l) + log bold(R)_(k j i)$ (valid on non-negative entries with $0 log 0 = 0$) and using $sum_i bold(R)_(k j i) = 1$ in the first resulting sum,
$ &-sum bold(L)_(j k l) bold(R)_(k j i) log (bold(L)_(j k l) bold(R)_(k j i)) \
  &quad = -sum bold(L)_(j k l) log bold(L)_(j k l) - sum_(k, j) (sum_l bold(L)_(j k l)) (sum_i bold(R)_(k j i) log bold(R)_(k j i)). $ <eq:entropy>
This is the chain rule for the entropy of a factorized distribution: the entropy of the joint over $(k, i)$ equals the entropy of the block choice plus the expected entropy of the within-block choice.

== Update for $bold(L)$

Fix $bold(R)$ and consider maximizing over one row $bold(L)_(j : l) in RR^m$. Collecting the terms of @eq:monarch-var and @eq:entropy that depend on it,
$ max_(bold(L)_(j : l)) quad &sum_(k, i, v) bold(L)_(j k l) bold(R)_(k j i) bold(Q)_((l j) v) bold(K)_((k i) v) - sum_k bold(L)_(j k l) log bold(L)_(j k l) - sum_k bold(L)_(j k l) (sum_i bold(R)_(k j i) log bold(R)_(k j i)) \
  "s.t." quad &bold(L)_(j k l) >= 0, quad sum_k bold(L)_(j k l) = 1. $
Grouping the linear terms as a single score per $k$,
$ max_(bold(L)_(j : l)) quad sum_k bold(L)_(j k l) underbrace((sum_v bold(Q)_((l j) v) sum_i bold(R)_(k j i) bold(K)_((k i) v) - sum_i bold(R)_(k j i) log bold(R)_(k j i)), bold(z)_k) - sum_k bold(L)_(j k l) log bold(L)_(j k l), $
which is exactly @eq:var with temperature $c = 1$. Hence
$ bold(L)_(j k l) = "softmax"_k (sum_v bold(Q)_((l j) v) sum_i bold(R)_(k j i) bold(K)_((k i) v) - sum_i bold(R)_(k j i) log bold(R)_(k j i)), $ <eq:L-update>
where $"softmax"_k$ is applied along $k$ with all other indices fixed. The score for block $k$ is the query's inner product with the $bold(R)$-weighted average key of that block, minus the negative entropy of the within-block distribution, so blocks with more spread-out attention receive a bonus.

== Update for $bold(R)$

Fix $bold(L)$ and consider one row $bold(R)_(k j :) in RR^b$. The $bold(L)$-entropy term is constant, and the remaining terms are
$ max_(bold(R)_(k j :)) quad &sum_(l, i, v) bold(L)_(j k l) bold(R)_(k j i) bold(Q)_((l j) v) bold(K)_((k i) v) - (sum_l bold(L)_(j k l)) (sum_i bold(R)_(k j i) log bold(R)_(k j i)) \
  "s.t." quad &bold(R)_(k j i) >= 0, quad sum_i bold(R)_(k j i) = 1. $
#block(breakable: false)[
Writing $c_(k j) = sum_l bold(L)_(j k l)$ and grouping the linear terms,
$ max_(bold(R)_(k j :)) quad sum_i bold(R)_(k j i) (sum_v bold(K)_((k i) v) sum_l bold(L)_(j k l) bold(Q)_((l j) v)) - c_(k j) sum_i bold(R)_(k j i) log bold(R)_(k j i), $
]
#block(breakable: false)[
which is @eq:var with temperature $c_(k j)$. When $c_(k j) > 0$,
$ bold(R)_(k j i) = "softmax"_i (1 / c_(k j) sum_v bold(K)_((k i) v) sum_l bold(L)_(j k l) bold(Q)_((l j) v)). $ <eq:R-update>
]
Here the score for key $i$ is its inner product with the $bold(L)$-weighted sum of the queries in grid column $j$ that route to block $k$, and the temperature is the total query mass $c_(k j)$ routed to that block. When $c_(k j) = 0$ every $bold(L)_(j k l) = 0$, the objective does not depend on $bold(R)_(k j :)$, and it may be left unchanged.

= Optimization

Write $f(bold(R))$ for the $bold(L)$-update @eq:L-update and $g(bold(L))$ for the $bold(R)$-update @eq:R-update. Both are exact maximizers of @eq:monarch-var over one factor with the other fixed. There are two natural ways to iterate them.

*Gauss-Seidel (alternating).* Each iteration uses the freshest factor,
$ bold(L)^(t+1) = f(bold(R)^t), quad bold(R)^(t+1) = g(bold(L)^(t+1)). $ <eq:gs>
This is block coordinate ascent with two blocks. Every update is an exact maximization over its block, so the objective is non-decreasing, and because the objective is bounded above on the constraint set the sequence of objective values converges. Per iteration, the two updates are sequential: the $bold(R)$-update cannot start until the $bold(L)$-update finishes.

*Jacobi (simultaneous).* Each iteration uses only the previous iterate,
$ bold(L)^(t+1) = f(bold(R)^t), quad bold(R)^(t+1) = g(bold(L)^t). $ <eq:jac>
The two updates are independent and can run in parallel. The price is that $(bold(L)^(t+1), bold(R)^(t+1))$ is not the result of any single exact block maximization, so the objective need not increase monotonically.

== Jacobi is two interleaved Gauss-Seidel chains

Substituting @eq:jac into itself,
$ bold(R)^(t+2) = g(f(bold(R)^t)), quad bold(L)^(t+2) = f(g(bold(L)^t)). $ <eq:jac2>
The map $g compose f$ is exactly one Gauss-Seidel iteration @eq:gs written in terms of $bold(R)$, and $f compose g$ is the same iteration written in terms of $bold(L)$ with the update order swapped. Hence the even Jacobi iterates $bold(R)^0, bold(R)^2, bold(R)^4, dots$ are precisely the Gauss-Seidel iterates started from $bold(R)^0$, and the odd Jacobi iterates $bold(R)^1, bold(R)^3, dots$ are the Gauss-Seidel iterates started from $bold(R)^1 = g(bold(L)^0)$. The same holds for $bold(L)$. Jacobi therefore runs two independent Gauss-Seidel chains, each advancing by one step every two Jacobi iterations.

Two consequences follow immediately.
- *Convergence.* Each chain inherits the monotone convergence of Gauss-Seidel. The Jacobi pair $(bold(L)^t, bold(R)^t)$ converges if and only if both chains reach the same fixed point; otherwise it settles into a period-two cycle between two stationary points, which Gauss-Seidel cannot do.
- *Rate.* If Gauss-Seidel contracts the objective gap by a factor $rho_"GS"$ per iteration near a fixed point, Jacobi contracts it by $rho_"GS"$ every two iterations, i.e. by
  $ rho_"J" = sqrt(rho_"GS") $ <eq:rate>
  per iteration. This is the classical relation $rho_"GS" = rho_"J"^2$ for two-block (consistently ordered) systems, here obtained without linearization because the two-block structure makes @eq:jac2 exact.

== Cost per iteration

Both schemes evaluate $f$ and $g$ once per iteration, so the arithmetic cost is identical: $Theta(N (m + b) d)$ per iteration by the same accounting as @eq:matmul. What differs is the critical path, measured in serial evaluations of $f$ or $g$. After $s$ serial evaluations, Gauss-Seidel holds one factor that has just been updated and one that is a step stale: at $s = 2T$ its $bold(L)$ was computed at depth $2T - 1$. Jacobi, if $f$ and $g$ are co-scheduled, completes $s$ iterations in the same depth, and since $bold(L)^s = f(bold(R)^(s-1))$ and $bold(R)^s = g(bold(L)^(s-1))$, _both_ of its factors are at full depth $s$. By @eq:rate the two schemes share the same asymptotic rate per unit depth, $rho_"J"^2 = rho_"GS"$, so this freshness advantage shows up as a better constant rather than a better rate, and it is largest when the depth budget is small. The extreme case is a budget of a single evaluation: Gauss-Seidel can only take a half step, $(f(bold(R)^0), bold(R)^0)$, which leaves $bold(R)$ at its data-independent initialization, whereas one Jacobi step $(f(bold(R)^0), g(bold(L)^0))$ adapts both factors to $bold(Q)$ and $bold(K)$. The price is twice the total work for a given depth, since by @eq:jac2 Jacobi is running a second Gauss-Seidel chain, and the latency advantage only materializes if the device has spare capacity to run $f$ and $g$ concurrently; a large batched update that already saturates the device gains nothing. Gauss-Seidel remains preferable when total work is the constraint and guarantees monotone ascent without two-cycles; Jacobi is preferable when latency is the constraint and parallel capacity is available, most clearly for very few steps.

== Numerical check

For $N = 256$, $m = b = 16$, $d = 32$ with Gaussian queries and keys scaled so that the scores have unit variance, and a shared random initialization, the objective gap to the converged value behaves as follows.

#figure(
  table(
    columns: 5,
    align: (right, right, right, right, right),
    table.header([Iteration], [GS gap], [GS KL], [Jacobi gap], [Jacobi KL]),
    [1], [$3.4 times 10^0$], [111.71], [$6.5 times 10^0$], [114.79],
    [2], [$7.9 times 10^(-2)$], [108.38], [$8.5 times 10^(-1)$], [109.15],
    [4], [$5.0 times 10^(-4)$], [108.30], [$3.9 times 10^(-2)$], [108.34],
    [6], [$9.3 times 10^(-6)$], [108.30], [$2.9 times 10^(-3)$], [108.31],
    [10], [$1.3 times 10^(-8)$], [108.30], [$3.7 times 10^(-5)$], [108.30],
    [20], [$< 10^(-12)$], [108.30], [$8.7 times 10^(-9)$], [108.30],
    [30], [$< 10^(-12)$], [108.30], [$3.9 times 10^(-12)$], [108.30],
  ),
  caption: [Objective gap and reverse KL to exact attention, summed over rows, for Gauss-Seidel (GS) and Jacobi iteration from the same initialization.],
)

A least-squares fit of the geometric rate gives $rho_"GS" approx 0.165$ and $rho_"J" approx 0.409$, so $rho_"J"^2 approx 0.168 approx rho_"GS"$, matching @eq:rate. Both schemes converge to the same fixed point here, and the Jacobi $bold(R)$-iterate at step $2t$ coincides with the Gauss-Seidel $bold(R)$-iterate at step $t$ to machine precision, confirming @eq:jac2. Comparing at equal depth rather than equal iteration count (Gauss-Seidel iteration $t$ against Jacobi iteration $2t$), Jacobi is ahead throughout, e.g. a gap of $8.5 times 10^(-1)$ versus $3.4 times 10^0$ at depth two and $3.9 times 10^(-2)$ versus $7.9 times 10^(-2)$ at depth four. At depth one, over 20 random draws of $bold(Q), bold(K)$, one Jacobi step beat the Gauss-Seidel half step in every draw, reducing the summed KL from 119.9 to 112.3 from a uniform initialization and from 213.5 to 129.1 from a random one. Note that a handful of Gauss-Seidel iterations already reach the KL floor of the Monarch class: the gap between the converged KL and the KL after two iterations is $10^(-2)$, whereas the KL floor itself is $10^2$, so in practice the approximation error is dominated by the Monarch constraint, not by incomplete optimization.

= Interpretation

The Monarch constraint models each attention row as a factorized distribution: from query $(l, j)$, first choose a key block $k$ with probability $bold(L)_(j k l)$, then a key $i$ within that block with probability $bold(R)_(k j i)$. Because $bold(R)_(k j i)$ does not depend on $l$, all queries in grid column $j$ share the same within-block key distributions but weight the blocks individually.

Using the identity $sum_n bold(p)_n bold(z)_n + H(bold(p)) = log sum_n e^(bold(z)_n) - "KL"(bold(p) parallel "softmax"(bold(z)))$ row by row and discarding the log-partition terms, which do not depend on $bold(P)$, @eq:monarch-var is equivalent to
$ min_(bold(L), bold(R)) sum_(l, j) "KL"(bold(P)_((l j) :) parallel bold(p)^*_(l j)), quad bold(p)^*_(l j) = "softmax"((bold(Q) bold(K)^top)_((l j) :)), $
so #ma finds the Monarch factorization closest to exact attention in reverse KL. Reverse KL is mode-seeking: it concentrates mass on the highest-attention keys.
