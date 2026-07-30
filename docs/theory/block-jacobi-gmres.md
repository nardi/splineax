# Block-Jacobi GMRES

## Introduction

Every other sparse solver in this package hands the matrix to an external library. That
works well on a CPU, but it means the solve cannot be part of a single compiled program, and
on a GPU or TPU it means the matrix has to leave the accelerator and come back. I wanted a
solver with no such boundary, one built only from array operations, so that a whole
optimization loop containing many solves compiles and runs as one program.

Doing without a library rules out the usual sparse direct factorizations. Those work by
building triangular factors of the matrix, and the order in which their rows are eliminated
depends on the values encountered along the way. That data-dependent control flow is exactly
what a compiled array program cannot express. An iterative method has the opposite shape: a
fixed sequence of matrix-vector products, which is a natural fit. The price is that an
iterative method needs a **preconditioner**, an approximate inverse cheap enough to apply on
every iteration and accurate enough to make the iteration converge quickly.

The solver described here builds such a preconditioner in three stages. The matrix is first
reordered to concentrate its nonzeros near the diagonal. The reordered index range is then
cut into equal, slightly overlapping blocks, and the corresponding diagonal blocks of the
matrix are inverted. Those inverses form the preconditioner, and a Krylov method uses it to
solve the reordered system. Each stage is a small number of array operations over index
vectors, which is what keeps the whole method inside one compiled program.

## Problem statement

The problem is to solve

$$ A x = b $$

for $x$, where $A$ is a square $n \times n$ matrix with only $\mathrm{nse}$ nonzero entries,
and $\mathrm{nse}$ is far smaller than $n^2$.

A direct method factors $A$ into triangular factors and then solves against them. The
difficulty is **fill-in**: the factors of a sparse matrix are generally much denser than the
matrix itself, and how much denser depends strongly on the elimination order. Controlling
fill-in is the central concern of a sparse direct solver, and it is why such solvers spend a
separate analysis stage searching for a good order before touching any values.

An iterative method avoids factoring altogether. A **Krylov subspace method** builds the
space spanned by $b, Ab, A^2 b, \ldots$ and takes the best approximation to $x$ available in
it, so the only thing it ever asks of $A$ is the ability to multiply a vector. The method used
here is the **generalized minimal residual** method (GMRES), due to
[Saad and Schultz](#ref-saad-schultz), which chooses the approximation minimising the residual
norm $\lVert b - A x \rVert$ over that space, and which places no symmetry or definiteness
requirement on $A$. In its restarted form the space is capped at a fixed dimension and rebuilt
from the current approximation, which bounds memory.

How fast a Krylov method converges depends on the spectrum of $A$, and for most matrices
arising from discretised problems it converges too slowly to be useful on its own. The remedy
is to supply an operator $M \approx A^{-1}$, so that the iteration effectively works with
$M A$, whose spectrum is clustered near one. Choosing $M$ is a trade-off with no general
solution: a more accurate $M$ costs more to build and more to apply on every iteration, and
the exact inverse would be an admission that the whole exercise was unnecessary.
[Saad](#ref-saad) surveys the Krylov methods and the preconditioners they are paired with.

Two ways of storing a sparse matrix appear throughout what follows. The **coordinate** form
(COO) stores three vectors of length $\mathrm{nse}$, giving the row index, the column index
and the value of each nonzero entry. The **compressed sparse row** form (CSR) replaces the
row-index vector with a vector of $n + 1$ offsets marking where each row begins, which
requires the entries to be sorted by row and makes the entries of a given row contiguous.
Introducing both here lets the sections below describe each stage as an operation on index
vectors.

## Overview

The method has three stages, separated by what each one depends on.

The **reordering** stage sees only the sparsity pattern. It computes a symmetric permutation
that gathers the nonzeros of $A$ into a narrow band around the diagonal, and it does so by
treating the pattern as a graph and numbering its vertices so that adjacent ones end up
close together. Nothing about the values enters, and the permutation is valid for every
matrix sharing the pattern.

The **blocking** stage also sees only the pattern. It cuts the reordered index range into
equal, slightly overlapping intervals, choosing their size by measuring what proportion of
the nonzeros fall inside the resulting diagonal blocks. The output is a description of where
each stored entry belongs, which fixes the shape of everything computed later.

The **solve** stage is the only one that sees values. It fills the diagonal blocks, inverts
each of them, and hands the collection to a Krylov method as a preconditioner for the
reordered system. Since the blocks are independent, both the inversion and each application
of the preconditioner are sets of small dense operations with no ordering between them.

The reason for drawing the boundary after the second stage rather than the third is that a
great many problems present a sequence of matrices sharing one pattern, a Newton iteration
being the obvious example. For those, the first two stages run once.

The sections below take the stages in turn, then [algorithm 5](#algo5) states the whole
method in one place.

## Bandwidth reduction

The **bandwidth** of $A$ is the largest distance from the diagonal at which a nonzero
appears,

$$ \beta(A) = \max \{\, |i - j| \;:\; A_{ij} \neq 0 \,\} $$

and its **envelope** is the set of positions lying between the diagonal and the outermost
nonzero in each row, a finer measure that does not collapse to the single worst row. A matrix
of small bandwidth has all of its nonzeros in a narrow diagonal strip.

Bandwidth is not a property of the underlying problem. It is a property of how the unknowns
happen to be numbered. Renumbering them by a permutation $P$ replaces $A$ with

$$ A' = P A P^{\mathsf{T}}, \qquad A' x' = b', \quad x' = P x, \quad b' = P b $$

which is a similarity transform by an orthogonal matrix. The transformed system has the same
solution up to the renumbering, the same eigenvalues, and the same condition number. Only the
positions of the nonzeros change. This is what makes reordering free in a numerical sense: it
can be chosen purely to improve structure, with no effect on what is being solved.

In COO the transform is cheaper still, and worth spelling out because it explains why the
reordering costs so little. The permutation appears nowhere in the values. Applying the
inverse permutation elementwise to the stored row-index and column-index vectors relabels
every entry where it lies, and that relabelling *is* the similarity transform. The cost is
linear in $\mathrm{nse}$ and consists entirely of integer reads. Recovering CSR afterwards
requires sorting the relabelled index pairs, and because that sort is determined by the
pattern alone it is computed once and afterwards replayed as a reordered read of the values.

Two closely related operators are stored quite differently here. The permutation $P$ is never
built as a matrix. It is an index vector, its action on a vector is a reordered read, and
$P A P^{\mathsf{T}}$ is the relabelling just described, so materialising it would buy
nothing. The reordered matrix $A'$,
in contrast, is built and kept, even though the same effect could be had by permuting on
either side of every product with the original $A$. It is built because the iteration applies
it many times, and because holding it in the reordered layout is what turns a reduced
bandwidth into contiguous memory access. Reordering pays off twice, once by concentrating
nonzeros where the preconditioner can reach them and once by improving locality of the
matrix-vector product, and only the second payoff requires the reordered matrix to exist.

## Reordering

Finding the permutation that minimises bandwidth is NP-hard, so the reordering stage uses
heuristics. Both heuristics below treat the pattern of $A$ as a graph on $n$ vertices, with an
edge between $i$ and $j$ whenever $A_{ij}$ or $A_{ji}$ is nonzero.

That graph needs no separate construction. The COO index pair already is an edge list, and
reading each stored entry in both directions symmetrises it. Vertex degrees, the expansion of
a search frontier and products with the graph Laplacian are then all reductions grouped by
endpoint over that edge list. Nothing resembling an adjacency list or a linked structure is
required, which matters because such structures are what make graph algorithms awkward to
express as array operations.

### Level-set ordering

The classical heuristic is **reverse Cuthill-McKee** (RCM): number the vertices in
breadth-first order, following [Cuthill and McKee](#ref-cuthill-mckee), then reverse the
result, which is [George and Liu](#ref-george-liu)'s addition.

> <span id="algo1"></span>**Algorithm 1** (level-set ordering)
  1. Choose an unvisited vertex of minimum degree as the seed, and assign it level $0$.
  2. Given the set of vertices at level $\ell$, find every unvisited neighbour of that set and
     assign it level $\ell + 1$. Record for each newly reached vertex the lowest-numbered
     vertex at level $\ell$ adjacent to it, as its parent.
  3. Repeat step 2 until no vertex is reached. If unvisited vertices remain, the graph is
     disconnected, so return to step 1 and seed a new component.
  4. Sort all vertices by level, then by parent, then by degree. Reverse the result.

The reason breadth-first search (BFS) reduces bandwidth is a counting argument. Once vertices
are numbered by level, an edge can only join vertices in the same or adjacent levels, since a
vertex two levels away is by definition not a neighbour. The bandwidth of the reordered matrix
is therefore bounded by the largest number of vertices in any two consecutive levels. Making
the levels thin makes the band narrow, and this bound holds no matter how vertices are ordered
within a level. Ordering within a level, by parent and then by degree, is a refinement that
improves the envelope in practice rather than the source of the guarantee. Reversal at the end
leaves the bandwidth unchanged but never increases the envelope, which is
[George and Liu](#ref-george-liu)'s observation and the reason the reversed form is preferred.

Expressing this as array operations is straightforward, with one caveat. Each round of step 2
is a pair of reductions over the edge list, cheap and independent of $n$ beyond the edge count,
and step 4 is a single sort. The caveat is that the number of rounds equals the number of
levels, which is a property of the graph rather than something that can be bounded in advance.
For a graph shaped like a long path the level count is of order $n$, and the ordering then
costs far more than the solve it is meant to accelerate. Such matrices are already narrowly
banded, so little is lost by leaving them alone, but the cost is a real limitation of this
heuristic.

### Spectral ordering

The second heuristic, due to [Barnard, Pothen and Simon](#ref-barnard), avoids the level-count
problem by replacing the search with an eigenvector computation. Let $L = D - W$ be the
**graph Laplacian**, where $W$ is the
adjacency matrix of the symmetrised pattern and $D$ the diagonal matrix of vertex degrees. $L$
is symmetric and positive semidefinite, and its smallest eigenvalue is zero with the constant
vector as eigenvector. The eigenvector of the second smallest eigenvalue is the **Fiedler
vector**.

> <span id="algo2"></span>**Algorithm 2** (spectral ordering)
  1. Compute the eigenvectors of the few smallest eigenvalues of $L$.
  2. Discard the near-constant vector.
  3. For each remaining eigenvector, sort the vertices by its entries to obtain a candidate
     permutation, and measure the bandwidth that candidate achieves.
  4. Keep the candidate of smallest bandwidth.

The intuition for step 3 is that the Fiedler vector is the smoothest non-trivial function on
the graph, in the sense that it minimises $\sum_{(i,j)} (v_i - v_j)^2$ over unit vectors
orthogonal to the constant. Adjacent vertices therefore receive similar values, so sorting by
those values places connected vertices close together, which is precisely what small bandwidth
requires. More formally, the same minimisation is a continuous relaxation of the discrete
problem of minimising $\sum_{(i,j)} (\pi_i - \pi_j)^2$ over permutations $\pi$, a quantity
closely related to the envelope.[^minsum]

[^minsum]: The discrete problem is the minimum 2-sum problem. The relaxation and the envelope
bound it gives are due to [Barnard, Pothen and Simon](#ref-barnard), who report envelope
reductions of more than a factor of two over level-set orderings on some matrices. That result
is for their full multilevel algorithm with a refinement pass, not for sorting by a single
eigenvector as in [algorithm 2](#algo2), which measures worse than the level-set ordering on
the patterns tested here.

Steps 1 to 4 exist rather than simply taking the Fiedler vector because the second eigenvalue
is not always simple. On a square grid, symmetry between the two coordinate directions makes it
exactly double, so the corresponding eigenspace is two-dimensional and contains no
distinguished vector: an eigensolver may legitimately return any vector from it, including a
diagonal combination that orders the grid badly. Disconnected graphs are worse still, since
there the eigenvalue zero has multiplicity equal to the number of components. Computing several
candidates and keeping the one that measurably does best converts both cases from a silent
failure into a choice, at a cost of one bandwidth evaluation per candidate. Component indicator
vectors, incidentally, turn out to be perfectly good orderings, because they keep each
component contiguous.

The Laplacian is the clearest implicit operator in the method. A block eigensolver such as
[Knyazev](#ref-knyazev)'s asks only for repeated products with a set of vectors, never for an
individual entry, so $L$ is never assembled. Its product is evaluated directly from the edge
list and the degree vector. Note also that only the pattern of $A$ enters this stage, never its
values, which is why the reordering can be computed once for a pattern and reused for every
matrix sharing it.

## Block preconditioning

The simplest preconditioner is **Jacobi**, which takes $M$ to be the inverse of the diagonal
of $A$. It costs almost nothing and ignores almost everything.

Extending it, let the index range be cut into contiguous groups and let $A_k$ denote the
diagonal block of $A'$ indexed by group $k$ in both dimensions. Taking

$$ M = \bigoplus_k A_k^{-1} $$

gives the **block-Jacobi** preconditioner. It captures every interaction between two unknowns
in the same group exactly, and no interaction between groups at all. Reordering is what makes
this worth doing: after bandwidth reduction the strongly interacting unknowns are close
together in the numbering, so contiguous groups capture most of the matrix.

It is worth being precise about what is discarded, because the two roles of $A'$ in the method
are easy to conflate. The entries outside every block are dropped from the preconditioner
only. GMRES multiplies by the full reordered matrix on every iteration, so the system being
solved is unchanged and the computed solution is a solution of the original problem. A poor
preconditioner costs iterations, never correctness. This is the essential difference between
approximating an inverse and approximating a matrix, and it is what makes an aggressive
approximation safe here.

Blocks that do not overlap nevertheless discard more than is necessary. An interaction between
two unknowns is captured only if both fall in the same group, so an unknown near a group
boundary loses most of its interactions regardless of how narrow the band is. Widening each
group so that it overlaps its neighbours recovers them. Overlap turns a group into a
**subdomain**, and the resulting family of methods are the **Schwarz** methods.

Overlap creates a difficulty of its own: a row belonging to several subdomains would have
several subdomains writing to it, and summing those contributions overcounts. The **restricted
additive Schwarz** (RAS) method of [Cai and Sarkis](#ref-cai-sarkis) resolves this by assigning
every row a single owning subdomain and discarding each subdomain's output outside the rows it
owns, which they found converges faster than the overcounting variant while sending less data.
Applying the preconditioner to a vector $y$ is then

> <span id="algo3"></span>**Algorithm 3** (restricted additive Schwarz apply)
  1. For each subdomain $k$, read the entries of $y$ indexed by $k$ to form $y_k$.
  2. Form $z_k = A_k^{-1} y_k$, for all $k$ independently.
  3. For each row, take its value from the single subdomain that owns it.

Steps 1 and 3 are a reordered read and a reordered write. Step 2 is a set of independent small
dense products, with no dependence between subdomains, which is the property that makes the
whole preconditioner cheap to apply and the reason a block method is preferred here to a
stronger preconditioner built on triangular factors.

The subdomain each stored entry belongs to, and its position inside that subdomain, follow
directly from its relabelled indices, so assembling every block is one pass over the stored
entries. Entries that no subdomain covers are given a destination outside the block array and
discarded there, which turns neglecting what falls outside the blocks into an addressing choice
rather than a test applied per entry.

How the preconditioner is stored divides at this point, and the division is the design. As an
$n \times n$ operator it is never assembled, not even sparsely: it exists only
as [algorithm 3](#algo3). Its individual blocks are made fully explicit, as small dense
inverses. An implicit outer operator costs no storage and no sparse arithmetic, while explicit
inner blocks reduce each application to independent dense products. The block-diagonal
approximation of $A'$ is likewise never formed as a sparse matrix, only as the collection of
blocks.

## Block size selection

Two parameters govern the partition: the block size $b$ and the fraction of it by which
consecutive blocks overlap. Blocks of size $b$ overlapping by $f b$ advance by a stride
$s = (1 - f) b$, so there are about $n / s$ of them.

The quantity to be traded against cost is the **capture fraction**, the proportion of the
nonzeros of $A'$ that fall inside some block. Capture is defined as a membership test applied
to each stored entry and averaged, so for a given pattern and partition it is measured rather
than estimated. That distinction matters, because the obvious estimate is misleading.

Consider non-overlapping blocks and an entry at offset $d = |i - j|$. The entry is captured
only when both of its indices fall in the same block, and of the $b$ positions a block spans,
only $b - d$ admit both. The chance of capture is therefore about

$$ 1 - \frac{d}{b} $$

which decays across the whole width of the block rather than holding until $d$ reaches $b$.
Setting $b$ to the value below which most of the offsets lie retains a good deal less than
most of the entries, since every offset contributes a loss. Measuring capture directly avoids
reasoning about this at all, and it accounts for overlap automatically. Overlap improves matters
sharply: with a stride of $s$ an offset is captured whenever $d < b - s$, whatever the
alignment, so every offset up to $f b$ is captured in full.

Against capture stands cost. Storing the inverses takes $b^2$ values per block, and applying
them takes the same number of operations, so both scale as

$$ \text{cost} \;\propto\; \frac{n}{s} b^2 \;=\; \frac{n b}{1 - f} $$

which grows with $b$. Larger blocks capture more and cost more, and the selection rule
therefore takes the cheapest partition that reaches a required capture:

> <span id="algo4"></span>**Algorithm 4** (block size selection)
  1. For each candidate $b$ up to a fixed maximum, measure the capture fraction and compute
     the cost.
  2. Discard candidates whose capture falls below the target.
  3. Among the rest, keep the one of least cost, preferring the smaller $b$ on a tie.
  4. If no candidate reaches the target, keep the largest permitted $b$.

Minimising cost rather than $b$ matters for small matrices, where the two differ. When $n$ is
below the permitted maximum, one block spanning the whole matrix captures everything, and it
costs $n^2$. A slightly smaller block reaching the same capture target would need two blocks
and cost more, so the single block wins. In that regime the preconditioner is the exact inverse
of $A'$ and GMRES converges in one iteration, which is to say the solver degenerates gracefully
into a dense direct method. For large sparse matrices the number of blocks is large, cost grows
monotonically with $b$, and the rule reduces to taking the smallest $b$ that meets the target.

The capping of $b$ is what keeps this from becoming a dense solve, and it interacts with block
inversion below, since the cost of inverting a block grows faster than the cost of storing it.
When convergence is poor, raising the overlap fraction is usually the better first move, as it
increases capture at a cost linear in $1/(1-f)$ rather than the steeper growth that comes from
raising the capture target and with it $b$.

A limitation of measuring capture on the pattern is that it counts entries without regard to
magnitude. The stage runs before any values are available, which is what allows its results to
be reused across matrices sharing a pattern, but it means a matrix whose largest entries sit
far from the diagonal is served worse than the capture fraction suggests.

## Block inversion

The blocks are inverted explicitly rather than factored and kept as factors. This is a
deliberate reversal of the usual advice, which is that forming an inverse is both slower and
less accurate than solving against a factorization, and it is justified by what the two choices
do to the shape of the computation. The same argument is made by
[Anzt and co-authors](#ref-anzt-batched), who build block-Jacobi preconditioners on graphics
processors this way for exactly this reason. Stored factors would make each application a
triangular
solve, which is sequential along the block: each unknown depends on those before it. An
explicit inverse makes each application a dense matrix-vector product, in which every output
depends on every input and nothing waits. Since the sole purpose of the block partition is to
produce work that can be done independently, reintroducing a sequential dependency inside each
block would defeat it.

Inverting one $b \times b$ block costs a small multiple of the $\tfrac{2}{3} b^3$ operations an
LU factorization takes, and which multiple depends on how much structure is exploited and how
much is asked of the result. Counting only leading terms, and taking LU as the
unit,[^flopcounts]

| Method | Flops relative to LU | Reveals rank |
| --- | --- | --- |
| Cholesky | $\tfrac{1}{2}$ | no, and needs a symmetric positive definite block |
| LU | $1$ | no |
| Householder QR | $2$ | no |
| Column-pivoted QR | $2$ | yes |
| SVD | more, and with a considerably larger constant | yes |

[^flopcounts]: Flop counts from Golub and Van Loan, [reference below](#ref-golub-van-loan).
Column pivoting shares the leading term of an unpivoted QR, since maintaining the column norms
it selects on is of lower order. Measured cost tells a rather different story than flop counts
do, because the two rank-revealing routines are the ones that block least well: timing batched
inversion here in double precision, with the batch sized to hold total work constant, pivoted
QR came out at 1.4, 2.0 and 7.0 times a batched LU inverse for block sizes 32, 64 and 128, and
the SVD at 4.2, 13.8 and 17.5. Those are wall-clock figures from one machine, so read them as
an indication of the ordering rather than as constants.

Because the total is $(n/s) \cdot O(b^3)$, or equivalently $O(n b^2 / (1-f))$, the choice of
method and the cap on $b$ are the two levers on the cost of this stage. That cost is paid on
every set of values, so it is the dominant cost when a sequence of related systems is solved,
as in a Newton iteration.

The reason to pay for a rank-revealing method is that a diagonal block can be singular even
when $A$ is not. A block containing a structurally zero diagonal, as arises in saddle-point
systems, is the standard example. An ordinary inverse of such a block does not exist, and a
computed one contains arbitrarily large entries that propagate into the iteration and destroy
it. Both the SVD and pivoted QR expose which directions of a block are numerically absent, the
former through small singular values and the latter through small entries on the diagonal of
its triangular factor.

Those directions are then left **unpreconditioned**: the inverse acts as the identity on them
rather than inverting them, and rather than sending them to zero as a pseudo-inverse would.
That choice looks minor and is not. Preconditioning is applied on the left, so the iteration
works with $M A x = M b$, and the solutions of that system coincide with those of $A x = b$
only when $M$ has no null space. Sending a direction to zero would put one there, and the
iteration could then converge, in good faith, to a vector that does not solve the original
system. Acting as the identity keeps $M$ invertible and costs nothing beyond leaving those
unknowns to the Krylov method to sort out.

The same reasoning bounds how small a direction may be before it is discarded. The threshold is
deliberately loose, around the square root of the working precision rather than the working
precision itself, which bounds the condition number of $M$ by its reciprocal. Inverting a
direction near the limit of the precision would gain nothing numerically and would make $M$ so
ill-conditioned that $M(b - Ax)$ ceases to track $b - Ax$, which matters because it is the
former that the iteration measures. A solve is therefore certified at the end against the
unpreconditioned residual, at the cost of one further product with $A$, so that a reported
convergence means convergence of the system that was asked about.

## Algorithm

Collecting the stages:

> <span id="algo5"></span>**Algorithm 5** (block-Jacobi GMRES)
  1. Form the symmetrised pattern of $A$ as an edge list, and compute a permutation by
     [algorithm 1](#algo1) or [algorithm 2](#algo2).
  2. Relabel the stored indices by the permutation, and choose a block size by
     [algorithm 4](#algo4).
  3. Determine, for each stored entry, which blocks contain it and where.
  4. Given values, assemble the diagonal blocks and invert each of them.
  5. Solve the reordered system with restarted GMRES, preconditioned by
     [algorithm 3](#algo3), and check the residual of the result against the tolerance.
  6. Undo the relabelling on the solution.

Steps 1 to 3 depend only on the pattern of $A$. Step 4 depends on its values, and step 5 on
the values and the right-hand side. That boundary is the reason the stages are separated: a
sequence of matrices sharing a pattern, which is what a Newton iteration or a parametric study
produces, needs steps 1 to 3 done once. The permutation, the block geometry, the per-entry
block destinations and the sort order restoring CSR are all fixed by the pattern, so all that
remains per matrix is assembling blocks, inverting them, and iterating.

The explicit and implicit operators, gathered in one place: the reordered matrix is explicit,
because it is applied many times and its layout carries the benefit of the reordering. The
permutation is implicit, being an index vector. The graph Laplacian is implicit, because only
its action is ever required. The preconditioner is implicit, its blocks explicit. The Krylov
basis is explicit, and being the one term whose size grows with the iteration count it is what
the restart parameter exists to bound.

## Extensions

Several directions would strengthen the method, in rough order of how much they would buy.

**Retaining the coupling between blocks.** A matrix of bandwidth at most $b$ is block
tridiagonal under a partition into blocks of size $b$, so keeping the sub- and superdiagonal
blocks alongside the diagonal ones discards nothing at all. A block LU sweep over such a system
is exact, which would make the preconditioner an exact inverse and GMRES converge in a single
iteration, turning the method into a direct banded solver. The cost is that the sweep is
sequential in the block index, where the present method is fully independent. This is the
territory of the [SPIKE algorithm](#ref-spike) of Polizzi and Sameh, whose reduced system is
designed precisely to recover parallelism in that sweep, and whose
[truncated form](#ref-truncated-spike) is the block-diagonal approximation used here. A
divide-and-conquer solution of the reduced system would reduce the sequential depth to
logarithmic in the number of blocks.

**Stronger preconditioners with parallel application.** An incomplete LU factorization of the
reordered matrix is a considerably better approximate inverse than a block-diagonal one, but
applying it requires triangular solves, which is the sequential structure this method avoids.
Two established remedies fit the same constraints as the present design: solving the triangular
systems approximately by a few Jacobi sweeps, and
[incomplete sparse approximate inverses](#ref-anzt-isai), which generalise block-Jacobi by
solving small local problems for the sparsity pattern of the inverse itself.

**Mixed and adaptive precision.** The preconditioner need not be as accurate as the matrix,
since its errors cost iterations rather than accuracy. Storing each block's inverse in a
precision chosen from that block's conditioning reduces both memory traffic and the cost of
applying it, which [Anzt and co-authors](#ref-anzt-precision) report leaves convergence
essentially unaffected.

**Better seeding for the level-set ordering.** [Algorithm 1](#algo1) seeds from a minimum-degree
vertex, a cheap substitute for a vertex of maximum eccentricity.
[George and Liu](#ref-george-liu)'s algorithm finds a better seed with one additional search per
component, and typically yields thinner levels and hence a narrower band.

**Multilevel spectral refinement.** [Algorithm 2](#algo2) sorts by an eigenvector directly,
which is the weakest form of the spectral approach. Coarsening the graph, ordering the coarse
version and refining the result back down is both faster and more accurate for large graphs, and
is how [Barnard, Pothen and Simon](#ref-barnard) actually use it.

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
