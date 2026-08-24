# Finite Solve Symplecticity Audit

The table audits the deployed finite fixed-point realization of the nonseparable GSC-HNN branch.

| System | Checkpoint | Samples | max rho_F | median iters | max epsilon_omega | max epsilon_fb |
|---|---:|---:|---:|---:|---:|---:|
| Double pendulum | `checkpoints/double_pendulum/gscd/multiseed/seed_0/best.pt` | 256 | 7.914e-10 | 10.0 | 1.329e-07 | 4.833e-09 |
| Spherical pendulum | `checkpoints/spherical_pendulum/gscd/multiseed/seed_0/best.pt` | 256 | 1.685e-11 | 7.0 | 1.489e-07 | 3.081e-11 |

Definitions: rho_F is the normalized implicit midpoint residual of the residual substep; epsilon_omega is the randomized symplectic form residual of the full terminated single step map; epsilon_fb is the forward backward residual of the full terminated single step map. Both map defects are unnormalized, matching the manuscript definitions.
