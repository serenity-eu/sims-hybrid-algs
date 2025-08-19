# SIMS Constraint Programming Model

## Constraint Model

Let $U = \{k_1, \ldots, k_n\} \subset \mathbb{N}$ be a set of $n$ parts of the area of interest, called the **universe**. This is a polygon partition of the area of interest — the parts do not overlap and together cover the full area.

Each satellite image $i$ is represented by a collection $P_i \subset U$ of parts. Let $I = \{P_1, \ldots, P_m\}$ be the set of all $m$ satellite images.

We aim to find a subset $T \subset \{1, \ldots, m\}$ of selected images that covers the entire area of interest:

$$
\bigcup_{i \in T} P_i = U \tag{1}
$$

While selecting all images trivially satisfies this constraint, we usually seek an optimal solution that minimizes a cost. Each image $i$ has an associated cost $W_i \in \mathbb{N}$, and the goal is to minimize:

$$
\min \sum_{i \in T} W_i \tag{2}
$$

This is the classical **weighted set cover** problem. Other objectives may also be incorporated:

### Resolution

Let $A_k \in \mathbb{N}$ be the area of part $k \in U$, and $R_i \in \mathbb{N}$ be the resolution of image $i$. We aim to minimize the sum of best (lowest) resolution for each part:

$$
\min \sum_{k \in U} \min \{ R_i \mid i \in T, k \in P_i \} \tag{3}
$$

### Incidence Angle

Let $F_i \in \mathbb{N}$ be the incidence angle of image $i$. We minimize the **maximum** angle among selected images:

$$
\min \max \{ F_i \mid i \in T \} \tag{4}
$$

### Cloud Coverage

Let $C_i \subset P_i$ be the cloudy parts of image $i$. Define:

$$
D_k := \{ i \in \{1, \ldots, m\} \mid k \in P_i \setminus C_i \}
$$

Let $V_k$ be a Boolean variable indicating whether part $k$ is **cloudy in the cover**:

$$
V_k \iff \bigwedge_{i \in D_k} (i \notin T) \tag{5}
$$

We then minimize the total area affected by clouds:

$$
\min \sum_{k \in U} V_k \cdot A_k \tag{6}
$$

This objective can also be turned into a constraint if the user requires that only covers below a certain cloud threshold are valid.

### Representation and Solvers

The model is linear and can also be encoded for Mixed Integer Linear Programming (MILP) solvers. The decision set $T$ is represented using binary variables $x_i \in \{0, 1\}$, where $x_i = 1$ means image $i$ is selected. In Constraint Programming (CP), this binary representation is necessary due to $T$ being of non-fixed cardinality.
