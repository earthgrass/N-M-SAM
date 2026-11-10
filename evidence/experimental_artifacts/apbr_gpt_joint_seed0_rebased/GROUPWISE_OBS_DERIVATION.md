# Groupwise OBS derivation

Let the current free coordinates have inverse-Hessian factor `Q = U^T U`. For a four-coordinate group, a candidate deletion pair is `D` and the still-free coordinates are `R`. Enforcing `w_D + delta_D = 0` in the local quadratic objective gives the minimum-norm constrained update

`delta_R = -Q_RD (Q_DD)^(-1) w_D`.

Using the factor directly, `Q_DD = U[:,D]^T U[:,D]` and `Q_RD = U[:,R]^T U[:,D]`, which yields the implemented update

`delta_R = -U[:,R]^T U[:,D] (U[:,D]^T U[:,D])^(-1) w_D`.

The same linear map is applied to every APBR perturbation probe. After the update, both deleted coordinates are set to exact zero and all four coordinates in the current group are frozen. Therefore the next group uses the trailing factor `U[i+4:,i+4:]`; retained coordinates in the current group participate in the present compensation but cannot be changed later.

For a deletion pair, the standard second-order deletion cost is `0.5 w_D^T R_D w_D`, where `R_D=(Q_DD)^(-1)`. Averaging the same quadratic form over `K` symmetric APBR probes adds `0.5/K sum_k epsilon_D(k)^T R_D epsilon_D(k)`. This is the expected second-order cost under the symmetric probe distribution; it is not a worst-case robust objective.

The implementation in `baseline_adapters/apbr_gpt_joint/adapter.py` evaluates all six pairs independently for each output row, uses lexicographic first-index tie resolution, performs the exact multi-coordinate update, propagates probes, zeros the selected pair, and freezes the group. Tests compare it with an explicit SPD constrained solve, including the case where different output rows choose different pairs. Original SparseGPT remains separate and unchanged; this algorithm is not claimed to be its bitwise replay.
