# Block-Jacobi GMRES

## Introduction

Every other sparse solver in this package hands the matrix to an external library. That works well
on a CPU, but it means the solve can't be part of a single compiled program, and on a GPU or TPU it
means the matrix has to leave the accelerator and come back. I wanted a solver without that
boundary, built only from array operations, so that a whole optimization loop containing many solves
compiles and runs as one program.

Doing without a library rules out the usual sparse direct factorizations. Those build triangular
factors of the matrix, and the order in which rows get eliminated depends on the values encountered
along the way. That data-dependent control flow is exactly what a compiled array program can't
express. An iterative method has the opposite shape, a fixed sequence of matrix-vector products,
which fits nicely. The price is that an iterative method needs a **preconditioner**, an approximate
inverse cheap enough to apply on every iteration and accurate enough to make the iteration converge
quickly.

So the question this page answers is: what preconditioner can we build out of nothing but array
operations? The answer here has three parts. We reorder the matrix to concentrate its nonzeros near
the diagonal, cut the reordered index range into equal overlapping blocks, and invert those blocks.
The inverses together form the preconditioner, and GMRES uses it to solve the reordered system. Each
part is a small number of array operations over index vectors, which is what keeps the whole method
inside one compiled program.

## Problem statement

We want to solve

$$ A x = b $$

for $x$, where $A$ is a square $n \times n$ matrix with only $\mathrm{nse}$ nonzero entries, and
$\mathrm{nse}$ is far smaller than $n^2$.

A direct method factors $A$ into triangular factors and then solves against them. The difficulty is
**fill-in**: the factors of a sparse matrix are generally much denser than the matrix itself, and
how much denser depends strongly on the elimination order. Controlling fill-in is the central
concern of a sparse direct solver, and it's why such solvers spend a separate analysis stage
searching for a good order before touching any values.

An iterative method avoids factoring altogether. A **Krylov subspace method** builds the space
spanned by $b, Ab, A^2 b, \ldots$ and takes the best approximation to $x$ available in it, so the
only thing it ever asks of $A$ is the ability to multiply a vector. The method used here is the
**generalized minimal residual** method (GMRES) of [Saad and Schultz](#ref-saad-schultz), which
picks the approximation minimizing the residual norm $\lVert b - A x \rVert$ over that space, and
which asks nothing about symmetry or definiteness of $A$. In its restarted form the space is capped
at a fixed dimension and rebuilt from the current approximation, which bounds memory.

How fast a Krylov method converges depends on the spectrum of $A$, and for most matrices coming out
of discretized problems it converges far too slowly to be useful on its own. The remedy is an
operator $M \approx A^{-1}$, so that the iteration effectively works with $M A$, whose spectrum is
clustered near one. Choosing $M$ is a trade-off with no general answer: a more accurate $M$ costs
more to build and more to apply on every iteration, and the exact inverse would be an admission that
the whole exercise was unnecessary. [Saad](#ref-saad) surveys the Krylov methods and the
preconditioners they get paired with.

Two ways of storing a sparse matrix show up throughout what follows. The **coordinate** form (COO)
stores three vectors of length $\mathrm{nse}$, giving the row index, the column index and the value
of each nonzero entry. The **compressed sparse row** form (CSR) replaces the row-index vector with a
vector of $n + 1$ offsets marking where each row begins, which requires the entries to be sorted by
row and makes the entries of a given row contiguous. Both are worth having in mind, because they're
what let us describe every stage below as an operation on index vectors.

## Overview

Before getting into any of the pieces, here is the shape of the whole thing. The work splits into
three phases, and what separates them is what each one needs to look at.

The first phase looks only at the **sparsity pattern**, never at a value:

> <span id="algo1"></span>**Algorithm 1** (symbolic phase)
>
> 1. Find a maximum matching of the pattern, which tells us which rows can be paired with which
>     columns.
> 2. If that matching doesn't cover every row, stop, because the matrix is then singular whatever
>     its values turn out to be.
> 3. Use the matching to decide how the unknowns should be renumbered, and reorder them to pull the
>     nonzeros into a narrow band around the diagonal.
> 4. Cut the reordered index range into equal overlapping blocks, picking a block size by measuring
>     how much of the matrix the blocks manage to capture.
> 5. Work out, for each stored entry, which blocks contain it and where inside them it lands.

The second phase is the first one that looks at values:

> <span id="algo2"></span>**Algorithm 2** (numeric phase)
>
> 1. Scatter the values into the diagonal blocks, using the destinations from step 5 above.
> 2. Invert each block, independently of all the others.

And the third does the actual solve:

> <span id="algo3"></span>**Algorithm 3** (solve phase)
>
> 1. Permute the right-hand side into the reordered numbering.
> 2. Run restarted GMRES on the reordered system, applying the block inverses as a preconditioner on
>     every iteration.
> 3. Undo the renumbering on the solution, and check the residual of the result against the
>     tolerance that was asked for.

Why split it up this way instead of doing everything in one go? Because a great many problems hand
you a whole *sequence* of matrices that share a single pattern, a Newton iteration being the obvious
example. For those, phase one runs once and is then reused, and if only the right-hand side changes
between solves then phase two is reused as well. Drawing the boundaries at "what does this step need
to know" is what makes that reuse possible at all.

The rest of this page takes the pieces in turn. It starts with steps 3 and 4 of the symbolic phase,
since those are the core of the method, and comes back to the matching in steps 1 and 2 afterwards,
because that's a refinement handling a case the core doesn't manage on its own.

## Bandwidth reduction

The **bandwidth** of $A$ is the largest distance from the diagonal at which a nonzero shows up,

$$ \beta(A) = \max \{\, |i - j| \;:\; A_{ij} \neq 0 \,\} $$

and its **envelope** is the set of positions lying between the diagonal and the outermost nonzero in
each row, a finer measure that doesn't collapse to the single worst row. A matrix of small bandwidth
has all of its nonzeros in a narrow diagonal strip.

The useful thing about bandwidth is that it isn't a property of the underlying problem at all. It's
a property of how we happened to number the unknowns. Renumbering them by a permutation $P$ replaces
$A$ with

$$ A' = P A P^{\mathsf{T}}, \qquad A' x' = b', \quad x' = P x, \quad b' = P b $$

which is a similarity transform by an orthogonal matrix. The transformed system has the same
solution up to the renumbering, the same eigenvalues, and the same condition number. Only the
positions of the nonzeros change. This is what makes reordering free in a numerical sense: we can
choose it purely to improve structure, without changing what is being solved.

In COO it's cheaper still, and worth spelling out because it explains why the reordering costs so
little. The permutation appears nowhere in the values. Applying the inverse permutation elementwise
to the stored row-index and column-index vectors relabels every entry with where it now lies, and
that relabeling *is* the similarity transform. The cost is linear in $\mathrm{nse}$ and consists
entirely of integer reads. Recovering CSR afterwards needs the relabeled index pairs sorted, and
because that sort is determined by the pattern alone we compute it once and afterwards replay it as
a reordered read of the values.

Two closely related operators get stored quite differently here, which is worth pausing on. The
permutation $P$ is never built as a matrix. It's an index vector, its action on a vector is a
reordered read, and $P A P^{\mathsf{T}}$ is the relabeling just described, so materialising it
would buy nothing. The reordered matrix $A'$, on the other hand, is built and kept, even though we
could get the same effect by permuting on either side of every product with the original $A$. It's
built because the iteration applies it many times, and because holding it in the reordered layout is
what turns a reduced bandwidth into contiguous memory access. So reordering pays off twice, once by
concentrating nonzeros where the preconditioner can reach them and once by improving locality of the
matrix-vector product, and only the second payoff needs the reordered matrix to actually exist.

## Reordering

Finding the permutation that minimizes bandwidth is NP-hard, so we use heuristics. Both of the ones
available here treat the pattern of $A$ as a graph on $n$ vertices, with an edge between $i$ and $j$
whenever $A_{ij}$ or $A_{ji}$ is nonzero.

That graph needs no separate construction, which is the main reason this works as array code at all.
The COO index pair already *is* an edge list, and reading each stored entry in both directions
symmetrizes it. Vertex degrees, the expansion of a search frontier and products with the graph
Laplacian are then all reductions grouped by endpoint over that edge list. Nothing resembling an
adjacency list or a linked structure is needed anywhere, which matters because those are exactly
what make graph algorithms awkward to write as array operations.

### Level-set ordering

The default heuristic is **reverse Cuthill-McKee** (RCM): number the vertices in breadth-first order
starting from a low-degree seed, order within each level by parent and then by degree, and reverse
the result at the end. It's due to [Cuthill and McKee](#ref-cuthill-mckee), with the reversal
[George and Liu](#ref-george-liu)'s addition, and both describe the procedure in full, so I won't
repeat it here.

What is worth understanding is *why* a breadth-first numbering narrows the band, because that also
tells us when it will fail to. Once vertices are numbered by level, an edge can only join vertices
in the same or in adjacent levels, since a vertex two levels away isn't a neighbor by definition.
So the bandwidth of the reordered matrix is bounded by the largest number of vertices in any two
consecutive levels. Thin levels give a narrow band, and notice that this bound holds no matter how
we order the vertices *within* a level. Ordering within a level is a refinement that improves the
envelope in practice, not the source of the guarantee. The reversal at the end leaves bandwidth
untouched but never increases the envelope, which is George and Liu's observation and the reason the
reversed form is the one everybody uses.

There is one caveat when we express this as array operations. Each round of the frontier expansion
is a pair of reductions over the edge list, cheap and independent of $n$ beyond the edge count, and
the final ordering is a single sort. But the number of *rounds* equals the number of levels, and
that's a property of the graph rather than something we can bound ahead of time. For a graph shaped
like a long path the level count is of order $n$, and the ordering then costs far more than the
solve it was supposed to accelerate. Matrices like that are already narrowly banded so little is
lost by leaving them alone, but it's a genuine limitation of this heuristic.

### Spectral ordering

The alternative, due to [Barnard, Pothen and Simon](#ref-barnard), sidesteps the level-count problem
by replacing the search with an eigenvector computation. Let $L = D - W$ be the **graph Laplacian**,
where $W$ is the adjacency matrix of the symmetrized pattern and $D$ the diagonal matrix of vertex
degrees. $L$ is symmetric and positive semidefinite, its smallest eigenvalue is zero with the
constant vector as eigenvector, and the eigenvector of the second smallest eigenvalue is the
**Fiedler vector**. Sorting the vertices by its entries gives the ordering.

Why should that work? The Fiedler vector is the smoothest non-trivial function on the graph, in the
sense that it minimizes $\sum_{(i,j)} (v_i - v_j)^2$ over unit vectors orthogonal to the constant.
Adjacent vertices therefore receive similar values, so sorting by those values places connected
vertices close together, which is precisely what a small bandwidth asks for.[^minsum]

[^minsum]: More formally, the same minimization is a continuous relaxation of the
discrete problem of minimizing $\sum_{(i,j)} (\pi_i - \pi_j)^2$ over permutations $\pi$, known as
the minimum 2-sum problem, a quantity closely related to the envelope. The relaxation and the
envelope bound it gives are due to [Barnard, Pothen and Simon](#ref-barnard), who report envelope
reductions of more than a factor of two over level-set orderings on some matrices. That result is
for their full multilevel algorithm with a refinement pass, not for sorting by a single eigenvector
as here, which measures worse than the level-set ordering on the patterns I tested.

One wrinkle is worth mentioning, because working around it shaped the implementation. The second
eigenvalue isn't always simple. On a square grid, symmetry between the two coordinate directions
makes it exactly double, so the corresponding eigenspace is two-dimensional and contains no
distinguished vector: an eigensolver may quite legitimately hand back any vector from it, including
a diagonal combination that orders the grid badly. Disconnected graphs are worse still, since there
the eigenvalue zero has multiplicity equal to the number of components. So rather than trusting the
Fiedler vector, we compute the few smallest eigenvectors, discard the near-constant one, sort by
each of the rest to get a candidate ordering, measure the bandwidth each candidate actually
achieves, and keep the best one. That converts both failure modes from a silent problem into a
choice, at a cost of one bandwidth evaluation per candidate. Component indicator vectors,
incidentally, turn out to be perfectly good orderings, because they keep each component contiguous.

The Laplacian is the clearest implicit operator in the method. A block eigensolver such as
[Knyazev](#ref-knyazev)'s only ever asks for repeated products with a set of vectors, never for an
individual entry, so $L$ is never assembled at all. Its product is evaluated straight from the edge
list and the degree vector.

## Block preconditioning

The simplest preconditioner is **Jacobi**, which takes $M$ to be the inverse of the diagonal of $A$.
It costs almost nothing and ignores almost everything.

We can do better without much more effort. Cut the index range into contiguous groups, and let $A_k$
denote the diagonal block of $A'$ indexed by group $k$ in both dimensions. Then

$$ M = \bigoplus_k A_k^{-1} $$

is the **block-Jacobi** preconditioner. It captures every interaction between two unknowns in the
same group exactly, and no interaction between groups at all. This is where the reordering earns its
keep: after bandwidth reduction the strongly interacting unknowns sit close together in the
numbering, so contiguous groups capture most of the matrix.

It's worth being precise about what we're discarding, because the two roles $A'$ plays here are easy
to conflate. The entries outside every block are dropped from the *preconditioner* only. GMRES
multiplies by the full reordered matrix on every iteration, so the system being solved is unchanged
and whatever we compute is a solution of the original problem. A poor preconditioner costs
iterations, never correctness. That's the essential difference between approximating an inverse and
approximating a matrix, and it's what makes such an aggressive approximation safe.

Blocks that don't overlap nevertheless throw away more than they need to. An interaction between two
unknowns is captured only when both fall in the same group, so an unknown sitting near a group
boundary loses most of its interactions no matter how narrow the band is. Widening each group so it
overlaps its neighbors recovers them. Overlap turns a group into a **subdomain**, and the resulting
family of methods are the **Schwarz** methods.

Overlap creates a difficulty of its own, though: a row belonging to several subdomains would have
several subdomains writing to it, and summing those contributions overcounts. The **restricted
additive Schwarz** (RAS) method of [Cai and Sarkis](#ref-cai-sarkis) resolves this by giving every
row a single owning subdomain and discarding each subdomain's output outside the rows it owns, which
they found converges faster than the overcounting variant while sending less data. Applying the
preconditioner to a vector $y$ is then

> <span id="algo4"></span>**Algorithm 4** (restricted additive Schwarz apply)
>
> 1. For each subdomain $k$, read the entries of $y$ indexed by $k$ to form $y_k$.
> 2. Form $z_k = A_k^{-1} y_k$, for all $k$ independently.
> 3. For each row, take its value from the single subdomain that owns it.

Steps 1 and 3 are a reordered read and a reordered write. Step 2 is a set of independent small dense
products with no dependence between subdomains, and that independence is the whole point. It's what
makes the preconditioner cheap to apply, and it's why a block method is preferred here over a
stronger preconditioner built on triangular factors.

Which subdomain a stored entry belongs to, and where it sits inside that subdomain, follow directly
from its relabeled indices, so assembling every block is one pass over the stored entries. Entries
that no subdomain covers get a destination outside the block array and are discarded there, which
turns "ignore what falls outside the blocks" into an addressing choice rather than a test applied
per entry.

How the preconditioner gets stored divides at this point, and that division is really the design. As
an $n \times n$ operator it is never assembled, not even sparsely: it exists only as [algorithm
4](#algo4). Its individual blocks, on the other hand, are made fully explicit as small dense
inverses. An implicit outer operator costs no storage and no sparse arithmetic, while explicit inner
blocks reduce each application to independent dense products. The block-diagonal approximation of
$A'$ is likewise never formed as a sparse matrix, only as the collection of blocks.

## Block size selection

Two parameters govern the partition: the block size $b$ and the fraction of it by which consecutive
blocks overlap. Blocks of size $b$ overlapping by $f b$ advance by a stride $s = (1 - f) b$, so
there are about $n / s$ of them.

What we're trading against cost is the **capture fraction**, the proportion of the nonzeros of $A'$
that fall inside some block. Capture is defined as a membership test applied to each stored entry
and averaged, so for a given pattern and partition it's *measured* rather than estimated. That
distinction matters more than it might seem, because the obvious estimate is misleading.

Here's why. Consider non-overlapping blocks and an entry at offset $d = |i - j|$. The entry is
captured only when both of its indices land in the same block, and of the $b$ positions a block
spans, only $b - d$ admit both. So the chance of capture is about

$$ 1 - \frac{d}{b} $$

which decays across the whole width of the block rather than holding up until $d$ reaches $b$. Set
$b$ to the value below which most of the offsets lie and you retain a good deal less than most of
the entries, because every offset contributes a loss. Measuring capture directly saves us from
reasoning about this at all, and it accounts for overlap automatically. Overlap helps sharply, by
the way: with a stride of $s$ an offset is captured whenever $d < b - s$, whatever the alignment, so
every offset up to $f b$ is captured in full.

Against capture stands cost. Storing the inverses takes $b^2$ values per block, and applying them
takes the same number of operations, so both scale as

$$ \text{cost} \;\propto\; \frac{n}{s} b^2 \;=\; \frac{n b}{1 - f} $$

which grows with $b$. Larger blocks capture more and cost more, so the selection rule takes the
cheapest partition that reaches a required capture:

> <span id="algo5"></span>**Algorithm 5** (block size selection)
>
> 1. For each candidate $b$ up to a fixed maximum, measure the capture fraction and compute the
>     cost.
> 2. Discard candidates whose capture falls below the target.
> 3. Among the rest, keep the one of least cost, preferring the smaller $b$ on a tie.
> 4. If no candidate reaches the target, keep the largest permitted $b$.

Minimizing cost rather than $b$ matters for small matrices, where the two come apart. When $n$ is
below the permitted maximum, one block spanning the whole matrix captures everything and costs
$n^2$. A slightly smaller block reaching the same capture target would need two blocks and cost
more, so the single block wins. In that regime the preconditioner is the exact inverse of $A'$ and
GMRES converges in one iteration, which is to say the solver degenerates gracefully into a dense
direct method. For large sparse matrices the number of blocks is large, cost grows monotonically
with $b$, and the rule reduces to taking the smallest $b$ that meets the target.

Capping $b$ is what keeps this from turning into a dense solve, and it interacts with block
inversion below, since the cost of inverting a block grows faster than the cost of storing it. When
convergence is poor, raising the overlap fraction is usually the better first move, since it
increases capture at a cost linear in $1/(1-f)$ rather than the steeper growth that comes from
raising the capture target and with it $b$.

One disadvantage of measuring capture on the pattern is that it counts entries without regard to
magnitude. The stage runs before any values are available, which is exactly what lets its results be
reused across matrices sharing a pattern, but it does mean a matrix whose largest entries sit far
from the diagonal is served worse than the capture fraction suggests.

## Block inversion

The blocks are inverted explicitly rather than factored and kept as factors. This is a deliberate
reversal of the usual advice, which is that forming an inverse is both slower and less accurate than
solving against a factorization, and the justification is what the two choices do to the *shape* of
the computation. Stored factors would make each application a triangular solve, which is sequential
along the block, since each unknown depends on the ones before it. An explicit inverse makes each
application a dense matrix-vector product, in which every output depends on every input and nothing
waits. Since the sole purpose of the block partition is to produce work that can be done
independently, reintroducing a sequential dependency inside each block would rather defeat it. The
same argument is made by [Anzt and co-authors](#ref-anzt-batched), who build block-Jacobi
preconditioners on graphics processors this way for exactly this reason.

Inverting one $b \times b$ block costs a small multiple of the $\tfrac{2}{3} b^3$ operations an LU
factorization takes, and which multiple depends on how much structure is exploited and how much is
asked of the result. Counting only leading terms, and taking LU as the unit:[^flopcounts]

| Method | Flops relative to LU | Reveals rank |
| --- | --- | --- |
| Cholesky | $\tfrac{1}{2}$ | no, and needs a symmetric positive definite block |
| LU | $1$ | no |
| Householder QR | $2$ | no |
| Column-pivoted QR | $2$ | yes |
| SVD | more, and with a considerably larger constant | yes |

[^flopcounts]: Flop counts from Golub and Van Loan, [reference
below](#ref-golub-van-loan). Column pivoting shares the leading term of an unpivoted QR, since
maintaining the column norms it selects on is of lower order. Measured cost tells a rather different
story than flop counts do, because the two rank-revealing routines are the ones that block least
well: timing batched inversion here in double precision, with the batch sized to hold total work
constant, pivoted QR came out at 1.4, 2.0 and 7.0 times a batched LU inverse for block sizes 32, 64
and 128, and the SVD at 4.2, 13.8 and 17.5. Those are wall-clock figures from one machine, so read
them as an indication of the ordering rather than as constants.

Because the total is $(n/s) \cdot O(b^3)$, or equivalently $O(n b^2 / (1-f))$, the choice of method
and the cap on $b$ are the two levers on the cost of this stage. That cost is paid on every set of
values, so it's the dominant cost when a sequence of related systems gets solved, as in a Newton
iteration.

Why pay for a rank-revealing method at all? Because a diagonal block can be singular even when $A$
isn't. A block containing a structurally zero diagonal, as arises in saddle-point systems, is the
standard example. An ordinary inverse of such a block doesn't exist, and a computed one contains
arbitrarily large entries that propagate into the iteration and destroy it. Both the SVD and pivoted
QR expose which directions of a block are numerically absent, the former through small singular
values and the latter through small entries on the diagonal of its triangular factor.

Those directions are then left **unpreconditioned**: the inverse acts as the identity on them,
rather than inverting them and rather than sending them to zero as a pseudo-inverse would. That
choice looks minor and really isn't. Preconditioning is applied on the left, so the iteration works
with $M A x = M b$, and the solutions of that system coincide with those of $A x = b$ only when $M$
has no null space. Sending a direction to zero would put one there, and the iteration could then
converge, in perfectly good faith, to a vector that doesn't solve the original system at all. Acting
as the identity keeps $M$ invertible and costs nothing beyond leaving those unknowns for the Krylov
method to sort out.

The same reasoning bounds how small a direction may be before we discard it. The threshold is
deliberately loose, around the square root of the working precision rather than the working
precision itself, which bounds the condition number of $M$ by its reciprocal. Inverting a direction
near the limit of the precision would gain nothing numerically, and it would make $M$ so
ill-conditioned that $M(b - Ax)$ stops tracking $b - Ax$, which matters because it's the former that
the iteration measures. So a solve is certified at the end against the *unpreconditioned* residual,
at the cost of one further product with $A$, so that a reported convergence means convergence of the
system that was actually asked about.

## Saddle-point systems

Block inversion left one case unresolved. A block whose diagonal is structurally zero has its
uninvertible directions left alone, which is safe but means the unknowns concerned never get
preconditioned at all. Those unknowns aren't some oddity, though. They're the defining feature of a
whole class of problems, so it's worth choosing the blocks differently so the case doesn't come up.

A **saddle-point system** is one where some unknowns carry no coefficient on their own diagonal,
because their equations are constraints rather than balances. Discretized incompressible flow is the
standard example. A velocity unknown appears in a Laplacian and so has a large diagonal entry, while
a pressure unknown appears only as the multiplier enforcing that the velocity field is
divergence-free, and no pressure-pressure term exists at all. Grouping the unknowns by kind gives

$$ A = \begin{bmatrix} F & B^{\mathsf{T}} \\ B & 0 \end{bmatrix} $$

Call a row **ordinary** when it has a stored diagonal entry, and a **constraint** row when it
doesn't. Renumbering alone can't help us here: a symmetric permutation sends the entry $(i,i)$ to
$(\pi_i, \pi_i)$, so it preserves exactly which rows carry a diagonal entry no matter how we order
the unknowns. What we *can* change is which unknowns share a block.

### The pattern as a bipartite graph

The reordering stage read the pattern as a graph on a single set of vertices, symmetrized so that an
entry and its transpose became one edge. Here we need the other reading. Take two disjoint sets of
vertices, one for the rows and one for the columns, and join row $i$ to column $j$ whenever $A_{ij}$
is stored. That's a **bipartite graph**, and the stored index pair is already its edge list, this
time with no symmetrization.

A **matching** is a set of edges no two of which share a vertex. Read as a table, that's a partial
one-to-one assignment of rows to columns in which every assigned pair is a stored entry. A matching
is **perfect** when it assigns every row, and **maximum** when no matching has more edges. A vertex
left unassigned is **free**.

The useful fact about matchings is that they grow along paths. An **alternating path** is one whose
edges lie alternately outside and inside the matching, and an **augmenting path** is an alternating
path that's free at both ends. Such a path holds one more edge outside the matching than inside it,
so if we exchange the two kinds along it we're left with a matching one edge larger. Berge's theorem
supplies the converse: a matching is maximum exactly when it admits no augmenting path. So "keep
finding augmenting paths until there aren't any" is a complete algorithm, and the only question is
how to find them efficiently.

### Finding a maximum matching

Augmenting one path at a time costs a search for every edge gained. [Hopcroft and
Karp](#ref-hopcroft-karp) observed that all *shortest* augmenting paths can be found in a single
search and exchanged together, and that the shortest length strictly increases after each such
round, which bounds the number of rounds by $O(\sqrt{n})$. Their paper has the analysis. What's
worth spelling out here is how the search gets expressed without any data-dependent control flow:

> <span id="algo6"></span>**Algorithm 6** (maximum bipartite matching)
>
> 1. Start from a matching built greedily, each free row taking a free column among its stored
>     entries, with competing claims settled by a fixed rule.
> 2. Search breadth-first from all free rows at once, stepping out along edges outside the matching
>     and back along edges inside it, recording the level at which each vertex is reached. Stop at
>     the first level holding a free column.
> 3. Walk back from those free columns through the levels, claiming vertices so as to form a maximal
>     set of augmenting paths sharing no vertex, again settling competing claims by a fixed rule.
> 4. Exchange matched and unmatched edges along every path found.
> 5. Repeat from step 2 until step 2 reaches no free column.

Steps 2 and 3 are the same frontier expansion the [level-set ordering](#level-set-ordering) uses,
run in each direction, so they're reductions grouped by endpoint over the edge list and need no
adjacency structure. Step 3 is the interesting substitution: it stands in for the depth-first search
of the original formulation, whose control flow depends on what it runs into and so can't be written
as array operations. That substitution costs us nothing, because the bound above only ever asks for
a *maximal* set of disjoint shortest paths per round, not a maximum one.

In practice step 1 leaves very little to do. On the patterns I measured, the greedy start comes
within a few edges of maximum and a single round of steps 2 to 4 finishes the job, so the matching
costs a handful of frontier sweeps, fewer than the reordering it accompanies.

### Blocks holding a constraint and its partner

Now for the reason we wanted the matching. Suppose it's perfect, and let constraint row $i$ be
assigned column $k$. That column has to be an ordinary one, because a constraint row has no entries
in constraint columns at all. Take any block holding both unknowns. Restricted to the two of them
the matrix reads

$$ \begin{bmatrix} A_{kk} & A_{ki} \\ A_{ik} & 0 \end{bmatrix},
\qquad \det = -A_{ik} A_{ki} $$

and $A_{ik}$ is nonzero because the matching chose it. In a saddle-point system the off-diagonal
blocks are transposes of one another, so $A_{ki}$ is nonzero as well and this submatrix is
invertible. The constraint row is no longer structurally empty inside its block, which is what made
such a block singular in the first place. This is the algebraic form of a construction due to
[Vanka](#ref-vanka), who built blocks from a single pressure cell together with the velocities on
its faces. Here the same pairing is read off the matrix instead of off the mesh.

Blocks are contiguous intervals of the reordered range, so all we have to do is place each
constraint beside its partner:

> <span id="algo7"></span>**Algorithm 7** (constraint-aware ordering)
>
> 1. Compute a bandwidth-reducing rank $r$ by one of the [reorderings](#reordering) above.
> 2. Compute a maximum matching by [algorithm 6](#algo6), assigning each constraint row $i$ a
>     partner column $p(i)$.
> 3. Give ordinary unknown $j$ the key $2 r_j$, and constraint unknown $i$ the key $2 r_{p(i)} + 1$.
> 4. Sort by key.

The doubling leaves a gap between consecutive ordinary unknowns, and the added one drops each
constraint into the gap immediately following its partner, leaving the order of the ordinary
unknowns among themselves untouched. The result is again a symmetric permutation, so everything said
under [bandwidth reduction](#bandwidth-reduction) still holds.

This ordering only departs from the plain one when the matrix really does have the shape above. The
test is that constraint rows have no entries in constraint columns, which is precisely the statement
that the lower right block is zero. A pattern failing that test is ordered as before.

### Structural rank

A maximum matching settles a second question for free, which is why step 2 of the [symbolic
phase](#algo1) can afford to reject a matrix outright. The determinant of $A$ is a sum over
permutations $\sigma$ of the products $\prod_i A_{i \sigma(i)}$, and a term can be nonzero only if
every factor in it is a stored entry, which is to say only if $\sigma$ is a perfect matching of the
pattern. So if no perfect matching exists, every term vanishes identically and $A$ is singular for
*every* assignment of values, not merely for unlucky ones. This is the Frobenius-König theorem, and
the size of a maximum matching is accordingly called the **structural rank**.

A pattern of deficient structural rank therefore poses a problem with no solution to find. The
solver reports it as an error during analysis instead of iterating, which is what it would otherwise
do, at length, before returning a residual that never falls.

### What this doesn't repair

Choosing the blocks well doesn't widen what a block method can see. Eliminating the ordinary
unknowns leaves the constraint unknowns coupled to one another through $B F^{-1} B^{\mathsf{T}}$,
and $F^{-1}$ is dense, so that coupling can reach clear across the problem. Where it does, no
partition into small blocks captures it and the iteration converges slowly however we choose the
blocks. The method leans on the constraints being local, which holds for a discretized problem and
fails for a pattern whose entries are scattered at random.

There's also a subtler gap between adjacency in the reordered range and adjacency inside the block
that ends up answering for a row. [Restricted additive Schwarz](#block-preconditioning) gives each
row a single *owning* block, and that block's window can end at the row itself, leaving its partner
one position outside it even though the two sit right next to each other in the ordering. A block
wide enough to keep its owned rows well clear of both edges, which is what a generous
`overlap_fraction` already buys, makes this rare. A block chosen too small for the pattern can make
it common instead, which is what happens when [algorithm 5](#algo5) is asked to measure a traced
pattern and falls back on an estimate sized for an average row rather than the pattern actually
given. Nothing breaks when it does happen: the rank-revealing block inverse is exactly the fallback
this leans on, so the affected rows end up only as preconditioned as they would have been without
the grouping. It's a reason to prefer measuring the block size eagerly, or setting one explicitly,
over trusting the estimate on a pattern with pronounced local structure. The
`reject_estimated_block_size` setting turns that preference into something the solver enforces
rather than something you have to remember, since with it set, reaching the estimate at all raises
instead of estimating.

### Provenance

Computing a matching and using it to decide which unknowns are held together is established practice
in sparse direct solvers for symmetric indefinite systems, where a nonsymmetric row permutation
would destroy the symmetry the factorization depends on. [Duff and Pralet](#ref-duff-pralet) use a
symmetric weighted matching to predefine $1 \times 1$ and $2 \times 2$ pivots ahead of ordering, and
[Schenk and Gärtner](#ref-schenk-gaertner) apply the same idea to highly indefinite systems.
[Hagemann and Schenk](#ref-hagemann-schenk) carry it over to preconditioning, ordering so that
matched entries form small diagonal blocks. Matching-driven grouping is also how several aggregation
multigrid methods coarsen, as in [D'Ambra, Filippone and Vassilevski](#ref-bootcmatch).

[Prokopenko and Tuminaro](#ref-prokopenko-tuminaro) build the same kind of block for a saddle-point
multigrid smoother, one per constraint unknown together with everything it touches on the mesh,
which is the overlapping-block idea [Vanka](#ref-vanka) started. To carry that pairing up their
multigrid hierarchy, though, they choose coarse unknowns by location on the Q2-Q1 mesh rather than
by a matching. That's faithful to the discretization they target, but it comes with no rank
guarantee the way a matching does, a gap the paper itself acknowledges.

So each ingredient here is well established. What I couldn't find written down anywhere is this
particular assembly, a structural matching used to group the blocks of an overlapping Schwarz
block-Jacobi preconditioner.

## Repairing an accidental diagonal

The guard in the previous section exists because a missing diagonal doesn't always mean a saddle
point. Take a diagonally dominant matrix and permute its rows alone, without permuting its columns
to match. The result has almost no diagonal entries left, yet nothing about the underlying problem
has changed. Grouping would be the wrong response here, and the guard declines it for exactly this
pattern, since the constraint-looking rows aren't free of entries in each other's columns. What this
case needs isn't a different preconditioner but a different renumbering, and one a symmetric
permutation can't supply, for the reason we already saw: $P A P^{\mathsf{T}}$ sends the diagonal
entry $(i,i)$ to $(\pi_i, \pi_i)$, so it can never turn an empty diagonal into a full one.

A **maximum matching** repairs it, using the same [algorithm 6](#algo6) we already have. When the
matching is perfect it assigns every row a distinct column, so reading it as a permutation of the
rows alone is exactly a **maximum transversal**, the classical device for moving large entries onto
the diagonal ahead of a direct factorization, due to [Duff and
Koster](#ref-duff-koster-1999)[^duff-koster-2001]. Row $i$ matched to column $\mathrm{partner}(i)$
becomes, after the transversal, row $\mathrm{partner}(i)$ of the permuted matrix, and its diagonal
entry there is exactly the matched one, nonzero by construction. Bandwidth reduction is then applied
afterward, to the pattern the transversal leaves behind, in the same way a direct solver orders
after finding its transversal rather than before.

[^duff-koster-2001]: The matching here is unweighted, choosing among several
rows that could fill a column by a fixed rule rather than by the size of the entry. [Duff and
Koster](#ref-duff-koster-2001)'s fuller treatment (and the MC64 codes built on it) weight the
matching to maximize the product of the chosen entries, which needs the values rather than the
pattern alone. That's a real difference in a way the next section comes back to, not just a
refinement left for later.

### What changes in the solve

A symmetric reordering moves a vector and restores a solution with the same permutation used both
ways, which is why the [solve phase](#algo3) reads `perm` for one and `inv_perm` for the other and
they happen to agree. A row permutation breaks that agreement. Reordering *equations* changes which
equation sits where, so it belongs on the right-hand side. It says nothing about what the *unknowns*
mean, so the solution is unaffected by it, and only the bandwidth-reducing part of the reordering
needs undoing. `perm` and `inv_perm` are computed accordingly, no longer as each other's inverse:
the first folds in the transversal, the second doesn't.

Transposing inherits this asymmetry rather than escaping it. For a symmetric reordering, transposing
the reordered matrix and transposing the original commute, so the same permutation serves both,
which is why a transpose has always been able to reuse an analysis unchanged. With a transversal
present that no longer holds, but the fix is one line rather than a fresh analysis. Reordering a
right-hand side and restoring a solution swap roles under a transpose, exactly as they would for any
linear system, so building the transposed pair is `perm ↦ inverse_permutation(inv_perm)` and
`inv_perm ↦ inverse_permutation(perm)`. Applied when the two already are each other's inverse, which
covers every pattern without a transversal, this reproduces them unchanged. So it's a strict
generalization of what the solver already did, rather than a special case bolted on beside it.

### What this repair does and doesn't guarantee

The bandwidth reduction that follows the transversal was chosen for the matrix it was given, which
is the row-permuted one, not its transpose. Solving the transposed system reuses that same choice
rather than computing a fresh one, which keeps a transpose cheap but doesn't promise it an equally
good ordering. The transposed solve is exact, only some of the time it needs substantially more of
the iteration budget than the forward solve on the same matrix does. A pattern where both directions
matter would be better served by a genuinely fresh analysis of the transposed pattern, which this
solver doesn't attempt.

A more basic limitation sits upstream of all that. An unweighted transversal has no way to prefer a
large, well-conditioned entry over a small, coincidental one when a row could fill a column several
ways, so on a matrix where the diagonal is one candidate among many similarly sized off-diagonal
ones, it can just as easily choose badly. And a row permutation is not a similarity transform,
unlike the symmetric reordering used everywhere else in this method, so it can leave a matrix's
eigenvalues, and with them how readily GMRES converges, far worse than the original despite touching
nothing about its condition number.

Both effects are mild when only a few rows are actually out of place, which is what an accidental
relabeling ordinarily looks like, and both grow with how much of the matrix a single transversal
has to move. Neither turns into a wrong answer, only into a slower one: the transversal only
permutes equations, so whatever the iteration converges to still solves the system it was asked to
solve. A matrix shuffled so thoroughly that its own locality is gone is, in that sense, no different
from any other pattern this method is a poor fit for, and it's addressed the same way [block size
selection](#block-size-selection) already is, by measuring rather than assuming.

## What gets stored

Gathering the explicit and implicit operators in one place, since which is which is most of the
design:

- The **reordered matrix** is explicit, because the iteration applies it many times and its layout
  carries the benefit of the reordering.
- The **permutation** is implicit, being an index vector.
- The **graph Laplacian** is implicit, because only its action is ever required.
- The **preconditioner** is implicit, existing only as [algorithm 4](#algo4), while its blocks are
  explicit.
- The **Krylov basis** is explicit, and being the one term whose size grows with the iteration
  count, it's what the restart parameter exists to bound.

## Extensions

Several directions would strengthen the method, in rough order of how much they'd buy.

**Retaining the coupling between blocks.** A matrix of bandwidth at most $b$ is block tridiagonal
under a partition into blocks of size $b$, so keeping the sub- and superdiagonal blocks alongside
the diagonal ones discards nothing at all. A block LU sweep over such a system is exact, which would
make the preconditioner an exact inverse and GMRES converge in a single iteration, turning the
method into a direct banded solver. The cost is that the sweep is sequential in the block index,
where the present method is fully independent. This is the territory of the [SPIKE
algorithm](#ref-spike) of Polizzi and Sameh, whose reduced system is designed precisely to recover
parallelism in that sweep, and whose [truncated form](#ref-truncated-spike) is the block-diagonal
approximation used here. A divide-and-conquer solution of the reduced system would bring the
sequential depth down to logarithmic in the number of blocks.

**Stronger preconditioners with parallel application.** An incomplete LU factorization of the
reordered matrix is a considerably better approximate inverse than a block-diagonal one, but
applying it needs triangular solves, which is the sequential structure this method set out to avoid.
Two established remedies fit the same constraints as the present design: solving the triangular
systems approximately by a few Jacobi sweeps, and [incomplete sparse approximate
inverses](#ref-anzt-isai), which generalize block-Jacobi by solving small local problems for the
sparsity pattern of the inverse itself.

**Mixed and adaptive precision.** The preconditioner needn't be as accurate as the matrix, since its
errors cost iterations rather than accuracy. Storing each block's inverse in a precision chosen from
that block's conditioning reduces both memory traffic and the cost of applying it, which [Anzt and
co-authors](#ref-anzt-precision) report leaves convergence essentially unaffected.

**Better seeding for the level-set ordering.** The [level-set ordering](#level-set-ordering) seeds
from a minimum-degree vertex, a cheap substitute for a vertex of maximum eccentricity. [George and
Liu](#ref-george-liu)'s algorithm finds a better seed with one additional search per component, and
typically yields thinner levels and hence a narrower band.

**Multilevel spectral refinement.** The [spectral ordering](#spectral-ordering) sorts by an
eigenvector directly, which is the weakest form of the spectral approach. Coarsening the graph,
ordering the coarse version and refining the result back down is both faster and more accurate for
large graphs, and is how [Barnard, Pothen and Simon](#ref-barnard) actually use it.

**A weighted transversal.** The [repair above](#repairing-an-accidental-diagonal) uses an unweighted
matching, which has no way to prefer a good diagonal candidate over a coincidental one. Weighting it
by entry magnitude, as MC64 does, would close that gap, at the cost of needing values and so of
moving the work out of the symbolic phase.

## References

- <span id="ref-cuthill-mckee"></span>E. Cuthill and J. McKee, *Reducing the bandwidth of
  sparse symmetric matrices*, Proc. 24th National Conference of the ACM, 1969.
- <span id="ref-george-liu"></span>A. George and J. W. H. Liu, *Computer Solution of Large
  Sparse Positive Definite Systems*, Prentice-Hall, 1981.
- <span id="ref-barnard"></span>S. T. Barnard, A. Pothen and H. D. Simon,
  [*A spectral algorithm for envelope reduction of sparse matrices*](https://ntrs.nasa.gov/api/citations/19970009822/downloads/19970009822.pdf),
  Numerical Linear Algebra with Applications 2(4), 1995.
- <span id="ref-golub-van-loan"></span>G. H. Golub and C. F. Van Loan, *Matrix Computations*,
  4th ed., Johns Hopkins University Press, 2013.
- <span id="ref-saad-schultz"></span>Y. Saad and M. H. Schultz, *GMRES: a generalized minimal
  residual algorithm for solving nonsymmetric linear systems*, SIAM Journal on Scientific and
  Statistical Computing 7(3), 1986.
- <span id="ref-saad"></span>Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
  SIAM, 2003.
- <span id="ref-cai-sarkis"></span>X.-C. Cai and M. Sarkis, *A restricted additive Schwarz
  preconditioner for general sparse linear systems*, SIAM Journal on Scientific Computing
  21(2), 1999.
- <span id="ref-anzt-batched"></span>H. Anzt, J. Dongarra, G. Flegar and
  E. S. Quintana-Ortí,
  [*Variable-size batched Gauss-Jordan elimination for block-Jacobi preconditioning on graphics processors*](https://www.sciencedirect.com/science/article/abs/pii/S0167819117302107),
  Parallel Computing 81, 2019.
- <span id="ref-anzt-precision"></span>H. Anzt, J. Dongarra, G. Flegar, N. J. Higham and
  E. S. Quintana-Ortí,
  [*Adaptive precision in block-Jacobi preconditioning for iterative sparse linear system solvers*](https://www.netlib.org/utk/people/JackDongarra/PAPERS/Anzt_et_al-2018-Concurrency.pdf),
  Concurrency and Computation: Practice and Experience 31(6), 2019.
- <span id="ref-anzt-isai"></span>H. Anzt, T. K. Huckle, J. Bräckle and J. Dongarra,
  [*Incomplete sparse approximate inverses for parallel preconditioning*](https://www.sciencedirect.com/science/article/abs/pii/S016781911730176X),
  Parallel Computing 71, 2018.
- <span id="ref-spike"></span>E. Polizzi and A. H. Sameh,
  [*A parallel hybrid banded system solver: the SPIKE algorithm*](https://www.sciencedirect.com/science/article/abs/pii/S0167819105001353),
  Parallel Computing 32(2), 2006.
- <span id="ref-truncated-spike"></span>M. Manguoglu, A. H. Sameh and O. Schenk,
  [*Analysis of the truncated SPIKE algorithm*](https://epubs.siam.org/doi/10.1137/080719571),
  SIAM Journal on Matrix Analysis and Applications 30(4), 2009.
- <span id="ref-knyazev"></span>A. V. Knyazev, *Toward the optimal preconditioned eigensolver:
  locally optimal block preconditioned conjugate gradient method*, SIAM Journal on Scientific
  Computing 23(2), 2001.
- <span id="ref-duff-koster-1999"></span>I. S. Duff and J. Koster,
  [*The design and use of algorithms for permuting large entries to the diagonal of sparse matrices*](https://www.semanticscholar.org/paper/The-Design-and-Use-of-Algorithms-for-Permuting-to-Duff-Koster/284605c1ffc8aa65b8bb3bdbc3a53e69c069cde8),
  SIAM Journal on Matrix Analysis and Applications 20(4), 1999.
- <span id="ref-duff-koster-2001"></span>I. S. Duff and J. Koster,
  [*On algorithms for permuting large entries to the diagonal of a sparse matrix*](https://epubs.siam.org/doi/10.1137/S0895479899358443),
  SIAM Journal on Matrix Analysis and Applications 22(4), 2001.
- <span id="ref-hopcroft-karp"></span>J. E. Hopcroft and R. M. Karp,
  [*An $n^{5/2}$ algorithm for maximum matching in bipartite graphs*](https://epubs.siam.org/doi/10.1137/0202019),
  SIAM Journal on Computing 2(4), 1973.
- <span id="ref-vanka"></span>S. P. Vanka,
  [*Block-implicit multigrid solution of Navier-Stokes equations in primitive variables*](https://www.sciencedirect.com/science/article/abs/pii/0021999186900082),
  Journal of Computational Physics 65(1), 1986.
- <span id="ref-duff-pralet"></span>I. S. Duff and S. Pralet,
  [*Strategies for scaling and pivoting for sparse symmetric indefinite problems*](https://www.numerical.rl.ac.uk/media/reports/duprRAL2004020.pdf),
  SIAM Journal on Matrix Analysis and Applications 27(2), 2005.
- <span id="ref-schenk-gaertner"></span>O. Schenk and K. Gärtner,
  [*On fast factorization pivoting methods for sparse symmetric indefinite systems*](https://etna.math.kent.edu/volumes/2001-2010/vol23/abstract.php?vol=23&pages=158-179),
  Electronic Transactions on Numerical Analysis 23, 2006.
- <span id="ref-hagemann-schenk"></span>M. Hagemann and O. Schenk,
  [*Weighted matchings for preconditioning symmetric indefinite linear systems*](https://epubs.siam.org/doi/10.1137/040615614),
  SIAM Journal on Scientific Computing 28(2), 2006.
- <span id="ref-bootcmatch"></span>P. D'Ambra, S. Filippone and P. S. Vassilevski,
  [*BootCMatch: a software package for bootstrap AMG based on graph weighted matching*](https://dl.acm.org/doi/10.1145/3190647),
  ACM Transactions on Mathematical Software 44(4), 2018.
- <span id="ref-prokopenko-tuminaro"></span>A. Prokopenko and R. S. Tuminaro,
  [*An algebraic multigrid method for Q2-Q1 mixed discretizations of the Navier-Stokes equations*](https://doi.org/10.1002/nla.2109),
  Numerical Linear Algebra with Applications 24(6), 2017.
