# Sphere Function with Various Algorithms

This example shows how to run various QD algorithms in pyribs with the sphere
linear projection benchmark that originated in
[Fontaine 2020](https://arxiv.org/abs/1912.02400). Specifically, we consider a
version of the benchmark with 100-dimensional solutions and a 2-dimensional
measure space. Below, we show the mean and standard deviation over 20 trials
when running each algorithm on this benchmark for 10,000 iterations.

| Algorithm           |               QD Score |       Coverage |
| :------------------ | ---------------------: | -------------: |
| map_elites          |  416,111.64 ± 2,681.37 |  50.72 ± 0.40% |
| line_map_elites     |  491,013.92 ± 1,118.74 |  60.44 ± 0.17% |
| cvt_map_elites      |  416,350.11 ± 3,942.42 |  50.67 ± 0.49% |
| line_cvt_map_elites |  490,335.42 ± 4,818.54 |  60.39 ± 0.58% |
| me_map_elites       | 533,477.98 ± 15,041.68 |  65.28 ± 2.05% |
| cma_me_imp          |  456,678.01 ± 2,882.10 |  55.84 ± 0.41% |
| cma_me_imp_mu       |  498,464.35 ± 2,214.86 |  61.11 ± 0.34% |
| cma_me_basic        | 398,192.00 ± 10,174.36 |  47.04 ± 1.30% |
| cma_me_rd           |  458,403.60 ± 2,739.42 |  56.21 ± 0.42% |
| cma_me_rd_mu        |  512,692.56 ± 4,070.15 |  63.78 ± 0.61% |
| cma_me_opt          |   57,517.77 ± 3,042.75 |   5.97 ± 0.33% |
| cma_me_mixed        |  458,827.27 ± 2,359.05 |  56.19 ± 0.35% |
| og_map_elites       |  387,800.00 ± 2,760.75 |  46.83 ± 0.41% |
| omg_mega            |      753,680.67 ± 3.68 | 100.00 ± 0.00% |
| cma_mega            |      753,834.86 ± 2.15 | 100.00 ± 0.00% |
| cma_mega_adam       |      753,880.12 ± 1.92 | 100.00 ± 0.00% |
| cma_mae             |  633,368.55 ± 1,504.43 |  81.02 ± 0.27% |
| cma_maega           |      753,834.72 ± 4.45 | 100.00 ± 0.00% |
| ns_cma              | 154,937.46 ± 10,344.01 |  19.19 ± 1.38% |
| nslc                |  509,331.09 ± 2,565.67 |  63.01 ± 0.35% |
| nslc_cma_imp        |  537,891.90 ± 3,242.39 |  67.98 ± 0.43% |
| dds                 | 344,447.87 ± 40,153.65 |  72.72 ± 3.21% |
| dds_kde_sklearn     | 325,402.21 ± 36,203.61 |  75.88 ± 3.15% |
| dms                 |  700,776.58 ± 8,525.05 |  95.99 ± 1.43% |

## sphere.py

This is the primary file, showing how to set up the sphere benchmark and run QD
algorithms on it.

{gh-badge}`https://github.com/icaros-usc/pyribs/blob/master/examples/sphere.py`

```{eval-rst}
.. literalinclude:: ../../examples/sphere.py
    :language: python
    :linenos:
```

## sphere_multirun.py

This script calls sphere.py to run each algorithm for multiple trials and
produce the benchmark results at the top of this page.

{gh-badge}`https://github.com/icaros-usc/pyribs/blob/master/examples/sphere_multirun.py`

```{eval-rst}
.. literalinclude:: ../../examples/sphere_multirun.py
    :language: python
    :linenos:
```
