# SIMS Mixed Integer Linear Programming Model

## Linear Programming Model

For the mixed integer linear programming (MILP) formulation, we use the same nomenclature as for the constraint model. We add the necessary variables to linearize the model.

### Coverage Constraint

To linearize the cover constraint (1), it's necessary to associate each image $P_i$ with a decision variable $x_i$ that equals 1 if the image is selected, and 0 otherwise. We rewrite the constraint as follows:

$$
\sum_{i:k \in P_i} x_i \ge 1, \quad \text{for all } k \in U \tag{8}
$$

The previous constraint guarantees that all parts are covered by at least one image.

### Cost Objective

To linearize the cost constraint (2), it's necessary to associate each image $P_i$ with an auxiliary variable $w_i$ representing the image's cost. The linear constraint can be written as:

$$
\min \sum_{P_i \in I} x_i w_i \tag{9}
$$

The constraints (8) and (9) are the classical constraints used for set covering problems.

### Resolution Objective

The resolution objective is a min-min problem, where the objective is to minimize the sum of the minimum resolution of each part. The minimum resolution of a part is the minimum resolution of the images that contain them and belong to a cover. We need to add an auxiliary decision variable $r_k$ representing the best resolution of part $k$ and a large constant $B$, which is bigger than the maximum image resolution. We also need to add auxiliary binary decision variables $z_{kj}$ for each image $P_j$ that contains $k$. For each part $k$, we define $L_k := \{i \in \{1, \ldots, m\} \mid k \in P_i\}$ as the set of all images containing part $k$. For each part $k$, we can now define a constraint for the values that the variables $z_{kj}$ can take.

$$
\sum_{k=1}^{|L_k|} z_{kj} = |L_k| - 1 \tag{10}
$$

The constraint above states that only one of the $z_{kj}$ variables can be 0; the rest must be 1. We define the minimum resolution of a part as $r_k$. With the following two constraints, we can linearize (3).

$$
r_k \ge (x_j R_j + B(1 - x_i)) - 2B z_{kj} \quad \text{for all } j \in L_k \tag{11}
$$

$$
\min \sum_{k \in U} r_k \tag{12}
$$

### Incidence Angle Objective

To linearize the incidence angle objective, we need to minimize an auxiliary variable $\text{maxf}$ that represents the maximum incidence angle of the images in the cover.

$$
\min \text{maxf} \ge x_i F_i, \quad \text{for all } i = 1, \ldots, m \tag{13}
$$

### Cloud Coverage Objective

To minimize the area of the clouds, we can model this as a partial set cover problem, where the universe $C = \{1, \ldots, c\}$ is formed by all the clouds, and the sets are the images that can cover the clouds. For each cloud $c_i$, we have a variable $y_i$ that is 1 if the cloud is covered or 0 otherwise, and $A_c$ indicates the area of the cloud. To maximize the covering of the cloudy areas, we will minimize the following expression:

$$
\min -\sum_{c \in C} y_c A_c \tag{14}
$$

This is subject to the following constraint, which forces $y_c$ to be 0 if any of the images that cover it is selected to cover the AOI.

$$
\sum_{i:c \in P_{ic}} x_i \ge y_c, \quad \text{for all } c \in C \tag{15}
$$