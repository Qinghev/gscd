# Multi-Initial-Condition Evaluation

Each row aggregates all stored test initial conditions for each seed-isolated checkpoint.
The seed-level columns average over initial conditions first, then report mean +/- std across training seeds.

| System | Model | Seeds | ICs/seed | Steps | MSE seed mean +/- std | Drift seed mean +/- std | MSE all IC mean +/- std | Drift all IC mean +/- std | Eval time/seed (s) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| double_pendulum | gscd | 5 | 10 | 1000 | 7.2400e+00 +/- 2.47e-01 | 1.3315e+00 +/- 4.33e-01 | 7.2400e+00 +/- 1.09e+01 | 1.3315e+00 +/- 2.69e+00 | 8.0825e+00 +/- 6.56e-02 |
| double_pendulum | gsympnet | 5 | 10 | 1000 | 9.4719e+00 +/- 8.44e-01 | 1.5064e+00 +/- 2.50e-01 | 9.4719e+00 +/- 8.71e+00 | 1.5064e+00 +/- 1.92e+00 | 5.0147e+00 +/- 1.18e-01 |
| double_pendulum | hnn | 5 | 10 | 1000 | 7.7964e+00 +/- 7.07e-01 | 1.1649e+00 +/- 3.53e-01 | 7.7964e+00 +/- 1.09e+01 | 1.1649e+00 +/- 2.29e+00 | 3.5688e+00 +/- 5.19e-02 |
| double_pendulum | hnn_implicit | 5 | 10 | 1000 | 7.6748e+00 +/- 6.83e-01 | 1.6643e+00 +/- 8.51e-01 | 7.6748e+00 +/- 1.08e+01 | 1.6643e+00 +/- 3.90e+00 | 5.1580e+00 +/- 1.18e-01 |
| double_pendulum | node | 5 | 10 | 1000 | 1.3495e+01 +/- 1.77e+00 | 7.5529e+00 +/- 9.30e+00 | 1.3495e+01 +/- 1.27e+01 | 7.5529e+00 +/- 2.67e+01 | 1.1421e+00 +/- 3.23e-02 |
| spherical_pendulum | gscd | 5 | 20 | 1000 | 7.3852e-01 +/- 1.05e-01 | 3.6192e+06 +/- 7.12e+06 | 7.3852e-01 +/- 2.32e-01 | 3.6192e+06 +/- 2.91e+07 | 1.4469e+01 +/- 1.58e-01 |
| spherical_pendulum | gsympnet | 5 | 20 | 1000 | 1.3784e+00 +/- 6.91e-01 | 1.7072e+06 +/- 3.81e+06 | 1.3784e+00 +/- 7.17e-01 | 1.7072e+06 +/- 1.18e+07 | 1.0032e+01 +/- 1.46e-01 |
| spherical_pendulum | hnn | 5 | 20 | 1000 | 7.9710e-01 +/- 1.79e-01 | 3.0752e+07 +/- 6.21e+07 | 7.9710e-01 +/- 2.97e-01 | 3.0752e+07 +/- 2.75e+08 | 6.9399e+00 +/- 1.36e-01 |
| spherical_pendulum | hnn_implicit | 5 | 20 | 1000 | 2.3472e+00 +/- 8.09e-01 | 7.4930e+08 +/- 1.62e+09 | 2.3472e+00 +/- 9.02e-01 | 7.4930e+08 +/- 7.22e+09 | 9.6169e+00 +/- 1.78e-01 |
| spherical_pendulum | node | 5 | 20 | 1000 | 1.7746e+00 +/- 1.40e+00 | 9.5440e+08 +/- 2.02e+09 | 1.7746e+00 +/- 2.01e+00 | 9.5440e+08 +/- 8.97e+09 | 2.2244e+00 +/- 2.31e-02 |
