We implement hierarchical clustering for pokemon data.

We perform clustering on the publicly available Pokemon stats. Each Pokemon is defined by a row in the data set. We represent a Pokemon's strength by two numbers: "x" and "y". "x"  represents the Pokemon's total offensive strength, which is defined by Attack + Sp. Atk + Speed. Similarly, "y" represents the Pokemon's total defensive strength, which is defined by Defense + Sp. Def + HP. After each Pokemon becomes that two-dimensional feature vector, we cluster the first 20 Pokemon with hierarchical agglomerative clustering.
