A practical way to parallelize PLS with an ND-Tree archive is to (1) parallelize *neighborhood exploration* aggressively, and (2) make archive access *read-mostly* and *batched*, so that threads rarely contend on the ND-Tree. Below are implementable designs, from simplest to highest throughput, plus the ND-Tree mechanics needed for low blocking.

References for context: ND-Tree update mechanics ([arXiv][1]) and prior work explicitly revisiting parallel PLS ([cmap.polytechnique.fr][2]).

---

## 1) Parallelize PLS without fighting the archive

### Core decomposition

PLS typically maintains:

* an **archive** `A` (your ND-Tree),
* an **unexplored set** `U ⊆ A` (solutions whose neighborhoods still need exploration).

Parallel version:

* `U` becomes a **concurrent work pool** (per-thread deques + work stealing).
* Each worker explores neighborhoods and produces **candidate points**.
* Candidates are filtered locally, then merged into the global archive in **batches**.

### Why batching matters

ND-Tree updates can delete dominated points and restructure nodes; doing that per-candidate turns your archive into a lock hotspot. Batching converts many small contended updates into a small number of larger, cache-friendly updates.

---

## 2) Architecture that performs well in practice

### Data structures

* `WorkDeques[T]`: per-thread Chase–Lev deques (or similar) + work stealing.
* `LocalBuffer[T]`: vector of candidate objective vectors + solution pointers.
* `LocalND[T]`: small per-thread nondominated filter (simple list or tiny ND-tree).
* `GlobalArchive`: ND-Tree (read-mostly).
* `Seen/Explored`: optional concurrent hash set / Bloom+hash for dedup of set-cover bitsets.

### Worker loop (high level)

1. Pop `s` from local deque; steal if empty.
2. Generate neighborhood `N(s)` (in parallel inside the thread if huge, or sequential with fast delta eval).
3. For each neighbor `x`:

   * compute objectives (incremental deltas for set cover are critical here),
   * **fast reject** using a snapshot of global archive bounds,
   * insert into `LocalND[T]` if not locally dominated.
4. Periodically flush:

   * move `LocalND[T]` content into `LocalBuffer[T]` batch,
   * submit batch to the global archive integration path,
   * receive “accepted” new archive points back → push them into some worker deque as unexplored.

---

## 3) Make ND-Tree access least-blocking: three implementation options

### Option A (simplest, good): single-writer “archive integrator”

**Idea:** all workers are readers; one dedicated thread performs ND-Tree updates.

**How it works**

* Workers never write the ND-tree.
* Workers enqueue candidate batches to an MPSC queue.
* The integrator thread:

  * pulls a batch,
  * applies ND-Tree updates sequentially (including deletions),
  * emits newly accepted solutions into the work pool.

**Pros**

* Almost zero blocking for workers.
* ND-Tree stays simple and correct (no concurrent mutation).
* Very good throughput if objective evaluation / neighborhood generation dominate.

**Cons**

* Integrator can bottleneck if candidate rate is extremely high.

This matches “least interruptive” best and is often the first thing that scales.

---

### Option B (higher throughput): RCU / persistent ND-Tree with atomic root swap (read-copy-update)

**Idea:** readers traverse immutable nodes; writers create a new path and CAS the root pointer.

**Mechanics**

* Store `root` as `atomic<Node*>`.
* Reader:

  * `r = root.load(acquire)`; traverse without locks.
* Writer (batch update):

  * `r0 = root.load()`
  * `r1 = apply_batch_persistent(r0, batch)` producing a *new* root (structural sharing)
  * `compare_exchange(root, r0, r1)`
  * on CAS failure, retry using the new root.

**Memory reclamation**
Use epoch-based reclamation (EBR) or hazard pointers so old versions are freed only after all readers are past the epoch.

**Pros**

* Reads are fully lock-free.
* Writes are low-contention if batched.
* Very scalable on read-heavy workloads.

**Cons**

* ND-Tree update must be implemented in a persistent (copy-on-write) style.
* Handling deletions (removing dominated points) is more complex: you’ll rebuild/copy affected leaf lists and bounding boxes.

This is the best “least blocking” multi-writer pattern if you can accept persistent-tree complexity.

---

### Option C (hardest): fine-grained locking inside ND-Tree

**Idea:** per-node RW locks or optimistic locks; lock coupling while descending.

**Pattern**

* Readers take no locks or optimistic version checks.
* Writers:

  * lock nodes along the update path (hand-over-hand),
  * possibly lock subtrees when deleting dominated points,
  * update bounds, split/merge nodes.

**Pros**

* No copying overhead.

**Cons**

* Dominated-point deletion can lock large parts of the tree.
* Easy to deadlock or degrade under contention.
* Typically worse than Option B once contention is non-trivial.

Unless you have strong reasons, Option B or A is usually better for ND-Tree.

---

## 4) Critical performance trick: two-phase filtering before touching ND-Tree

### Phase 1: very cheap global pruning using ND-Tree node bounds

ND-Tree nodes keep approximate ideal/nadir bounds for subsets ([arXiv][1]). Use them as a *read-only dominance filter*:

* If candidate `x` is dominated by a node’s (approx) **nadir** in a way that guarantees dominance by all points in that node (depending on how bounds are maintained), skip that subtree.
* If `x` dominates the node’s **ideal** (again depending on bound semantics), you can mark subtree “possibly removable” but avoid deletion in worker threads; defer to batch integrator.

Even if bounds are “approximate”, they still prune a lot without comparisons.

### Phase 2: per-thread local nondominated buffer

Keep `LocalND[T]` small (e.g., cap at 512–4096). Use:

* simple vector + linear dominance checks (4 objectives is cheap),
* or a tiny ND-tree / quad-tree variant.

Only survivors go into global batches.

This reduces global archive update traffic drastically.

---

## 5) Handling “unexplored” in parallel (avoid duplicate work)

Parallel PLS typically relaxes the strict “each archive point explored exactly once”; you still want to bound duplicates.

Practical approach:

* Each archive solution gets a stable `id` (hash of set-cover bitset).
* Maintain `Explored` as:

  * `concurrent_hash_set<id>` (exact), or
  * `bloom_filter` + occasional exact check.

When a solution is accepted into the archive, attempt `if Explored.insert(id) == true` then push to `U`; otherwise skip scheduling.

This keeps scheduling low-contention (hash set sharded) and avoids repeated neighborhood exploration.

---

## 6) ND-Tree update strategy for concurrency

### Batch update semantics

Given a batch `B`, integration does:

1. For each `x ∈ B`: query if dominated by archive; if yes discard.
2. For each non-dominated `x`: insert.
3. Remove any archive points dominated by inserted points.

Key point: **do deletions only in the integrator / writer context**, never in worker threads.

### Lazy deletion + periodic rebuild (recommended)

Deletion in a tree is the concurrency killer. Instead:

* Mark dominated points as **tombstones** in leaves (atomic flag).
* Keep per-leaf live count.
* When tombstone ratio exceeds a threshold (e.g., 30–50%), do a **rebuild**:

  * collect live points by traversing current root snapshot,
  * build a fresh ND-Tree offline,
  * atomically swap root (Option A: under exclusive lock; Option B: CAS root).

This matches known “reverse / rebuild” ideas used to reduce structural churn in ND-type archives. ([hisaolab-sustech.github.io][3])

---

## 7) Sharding (optional): when the archive itself is too hot

If you truly need multi-writer throughput beyond Options A/B:

* Partition objective space into `K` shards (e.g., by Morton code / grid on normalized objectives).
* Each shard has its own ND-Tree (and its own lock / writer).
* Maintain a small “shard summary” structure: per-shard ideal/nadir to quickly rule out dominated candidates and to decide which shards must be queried.

Caveat: exact dominance is global; you still need to check candidates against any shard that cannot be ruled out by shard bounds. With good normalization and K=8–64, this can scale well.

---

## 8) Concrete implementation plan (what to build first)

1. **Work stealing + local buffers** (independent of ND-Tree choice).
2. **Single-writer integrator** (Option A) with:

   * MPSC queue of batches,
   * ND-Tree updates + lazy tombstones,
   * periodic rebuild.
3. Measure:

   * candidate rate, integrator CPU%, ND-Tree comparisons.
4. If integrator saturates:

   * switch to **RCU persistent ND-Tree** (Option B) and keep batching.
5. Only then consider sharding if needed.

This path minimizes engineering risk while getting you close to peak performance.

---

## 9) What “least blocking” looks like in code terms (sketch)

* `root: atomic<Node*>`
* `enter_epoch()/exit_epoch()` around every archive query
* Worker:

  * `r = root.load()`
  * query-only traversal (no locks)
* Writer (batch):

  * build new root by applying updates to `r` (copy-on-write)
  * `CAS(root, r, r2)`; retry if fails
  * retire old root to epoch GC

Even with retries, batching keeps CAS frequency low.

---

If you tell me:

* target language (C++ / Rust),
* max archive size and neighborhood size distribution,
* whether you require strict PLS semantics or accept relaxed duplicates,

I can provide a concrete design with data layouts (node struct, leaf representation, epoch GC choice, and a batch update algorithm specialized for 4 objectives + set-cover bitset hashing).

[1]: https://arxiv.org/abs/1603.04798?utm_source=chatgpt.com "ND-Tree-based update: a Fast Algorithm for the Dynamic Non-Dominance Problem"
[2]: https://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2018/GECCO/proceedings/proceedings_files/pap465s3-file1.pdf?utm_source=chatgpt.com "Parallel Pareto Local Search Revisited - CMAP"
[3]: https://hisaolab-sustech.github.io/pdf/Reverse%20Strategy%20for%20Non-dominated.pdf?utm_source=chatgpt.com "Reverse Strategy for Non-dominated Archiving"
