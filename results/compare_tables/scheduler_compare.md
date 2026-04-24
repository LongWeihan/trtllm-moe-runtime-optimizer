# Collected Metrics

| Label | Requests | Batches | Avg Req/Batch | Avg Batch ms | Step Std ms | TTFT p90 s | E2E p90 s | TPOT p90 s | Throughput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_moe_default | 12 | 3 | 4.00 | 1597.79 | 233.71 | 0.0809 | 1.4786 | 0.0115 | 280.39 |
| hot_expert_default | 12 | 3 | 4.00 | 1911.58 | 202.50 | 0.0740 | 1.8421 | 0.0114 | 301.32 |
| hot_rank_default | 12 | 3 | 4.00 | 2068.25 | 203.79 | 0.0803 | 1.9107 | 0.0112 | 293.97 |
| mixed_burst_default | 12 | 3 | 4.00 | 1824.93 | 296.38 | 0.0878 | 1.9723 | 0.0141 | 263.02 |
| repeated_prefix_moe_default | 12 | 3 | 4.00 | 1848.56 | 195.04 | 0.0862 | 1.7533 | 0.0140 | 242.35 |
| balanced_moe_moe_v1_synthetic | 12 | 3 | 4.00 | 1583.38 | 188.14 | 0.0765 | 1.4860 | 0.0116 | 282.94 |
| hot_expert_moe_v1_synthetic | 12 | 12 | 1.00 | 1446.91 | 168.49 | 0.0118 | 1.5735 | 0.0099 | 99.52 |
| hot_rank_moe_v1_synthetic | 12 | 12 | 1.00 | 1527.66 | 168.47 | 0.0110 | 1.7298 | 0.0099 | 99.50 |
| mixed_burst_moe_v1_synthetic | 12 | 5 | 2.40 | 1613.81 | 172.82 | 0.0758 | 1.6103 | 0.0115 | 178.46 |
| repeated_prefix_moe_moe_v1_synthetic | 12 | 8 | 1.50 | 1309.78 | 182.60 | 0.0714 | 1.4198 | 0.0114 | 128.27 |
| balanced_moe_moe_v1_replay | 12 | 3 | 4.00 | 1596.47 | 193.80 | 0.0772 | 1.4861 | 0.0115 | 280.62 |
| hot_expert_moe_v1_replay | 12 | 12 | 1.00 | 1462.98 | 203.48 | 0.0107 | 1.5698 | 0.0098 | 98.43 |
| hot_rank_moe_v1_replay | 12 | 12 | 1.00 | 1518.91 | 165.84 | 0.0114 | 1.7123 | 0.0098 | 100.07 |
| mixed_burst_moe_v1_replay | 12 | 5 | 2.40 | 1615.38 | 170.40 | 0.0737 | 1.6128 | 0.0115 | 178.29 |
| repeated_prefix_moe_moe_v1_replay | 12 | 8 | 1.50 | 1309.67 | 188.16 | 0.0698 | 1.4127 | 0.0113 | 128.28 |
| balanced_moe_moe_v2_replay | 12 | 3 | 4.00 | 1465.10 | 140.22 | 0.0796 | 1.4541 | 0.0117 | 305.78 |
| balanced_moe_moe_v2_synthetic | 12 | 3 | 4.00 | 1432.33 | 109.55 | 0.0762 | 1.4699 | 0.0115 | 312.78 |
| hot_expert_moe_v2_replay | 12 | 6 | 2.00 | 1697.70 | 181.13 | 0.0660 | 1.7928 | 0.0112 | 169.64 |
| hot_expert_moe_v2_synthetic | 12 | 6 | 2.00 | 1698.75 | 178.38 | 0.0657 | 1.7917 | 0.0112 | 169.54 |
| hot_rank_moe_v2_replay | 12 | 12 | 1.00 | 1531.37 | 164.69 | 0.0115 | 1.7186 | 0.0100 | 99.26 |
| hot_rank_moe_v2_synthetic | 12 | 12 | 1.00 | 1526.94 | 166.65 | 0.0135 | 1.7165 | 0.0099 | 99.55 |
| mixed_burst_moe_v2_replay | 12 | 5 | 2.40 | 1561.32 | 168.58 | 0.0758 | 1.5660 | 0.0115 | 184.46 |
| mixed_burst_moe_v2_synthetic | 12 | 5 | 2.40 | 1527.62 | 106.46 | 0.0732 | 1.5340 | 0.0115 | 188.53 |
| repeated_prefix_moe_moe_v2_replay | 12 | 8 | 1.50 | 1283.00 | 109.51 | 0.0721 | 1.2848 | 0.0115 | 130.94 |
| repeated_prefix_moe_moe_v2_synthetic | 12 | 8 | 1.50 | 1279.33 | 113.60 | 0.0706 | 1.2756 | 0.0115 | 131.32 |
| hot_expert_baseline | 12 | 3 | 4.00 | 1911.58 | 202.50 | 0.0740 | 1.8421 | 0.0114 | 301.32 |
| hot_expert_patched | 12 | 6 | 2.00 | 1697.70 | 181.13 | 0.0660 | 1.7928 | 0.0112 | 169.64 |
| hot_rank_baseline | 12 | 3 | 4.00 | 2068.25 | 203.79 | 0.0803 | 1.9107 | 0.0112 | 293.97 |
| hot_rank_patched | 12 | 12 | 1.00 | 1531.37 | 164.69 | 0.0115 | 1.7186 | 0.0100 | 99.26 |

## Comparisons

| Baseline | Candidate | TTFT p90 delta s | E2E p90 delta s | TPOT p90 delta s | Step Std delta ms | Throughput delta tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced_moe_default | balanced_moe_moe_v1_synthetic | -0.0043 | 0.0074 | 0.0001 | -45.57 | 2.55 |
| balanced_moe_default | balanced_moe_moe_v1_replay | -0.0036 | 0.0075 | 0.0001 | -39.91 | 0.23 |
| balanced_moe_default | balanced_moe_moe_v2_synthetic | -0.0046 | -0.0087 | 0.0001 | -124.16 | 32.39 |
| balanced_moe_default | balanced_moe_moe_v2_replay | -0.0013 | -0.0245 | 0.0002 | -93.49 | 25.40 |
| hot_expert_default | hot_expert_moe_v1_synthetic | -0.0622 | -0.2686 | -0.0015 | -34.01 | -201.80 |
| hot_expert_default | hot_expert_moe_v1_replay | -0.0633 | -0.2723 | -0.0016 | 0.97 | -202.89 |
| hot_expert_default | hot_expert_moe_v2_synthetic | -0.0083 | -0.0504 | -0.0002 | -24.12 | -131.79 |
| hot_expert_default | hot_expert_moe_v2_replay | -0.0080 | -0.0493 | -0.0002 | -21.37 | -131.68 |
| hot_rank_default | hot_rank_moe_v1_synthetic | -0.0693 | -0.1809 | -0.0013 | -35.32 | -194.47 |
| hot_rank_default | hot_rank_moe_v1_replay | -0.0689 | -0.1983 | -0.0014 | -37.95 | -193.90 |
| hot_rank_default | hot_rank_moe_v2_synthetic | -0.0668 | -0.1942 | -0.0014 | -37.14 | -194.42 |
| hot_rank_default | hot_rank_moe_v2_replay | -0.0688 | -0.1921 | -0.0012 | -39.10 | -194.71 |
| mixed_burst_default | mixed_burst_moe_v1_synthetic | -0.0120 | -0.3620 | -0.0026 | -123.56 | -84.56 |
| mixed_burst_default | mixed_burst_moe_v1_replay | -0.0141 | -0.3595 | -0.0026 | -125.98 | -84.74 |
| mixed_burst_default | mixed_burst_moe_v2_synthetic | -0.0145 | -0.4383 | -0.0026 | -189.92 | -74.50 |
| mixed_burst_default | mixed_burst_moe_v2_replay | -0.0120 | -0.4063 | -0.0026 | -127.80 | -78.57 |
| repeated_prefix_moe_default | repeated_prefix_moe_moe_v1_synthetic | -0.0149 | -0.3335 | -0.0027 | -12.44 | -114.09 |
| repeated_prefix_moe_default | repeated_prefix_moe_moe_v1_replay | -0.0164 | -0.3406 | -0.0027 | -6.88 | -114.07 |
| repeated_prefix_moe_default | repeated_prefix_moe_moe_v2_synthetic | -0.0156 | -0.4777 | -0.0026 | -81.44 | -111.03 |
| repeated_prefix_moe_default | repeated_prefix_moe_moe_v2_replay | -0.0141 | -0.4685 | -0.0025 | -85.53 | -111.41 |
| hot_expert_baseline | hot_expert_patched | -0.0080 | -0.0493 | -0.0002 | -21.37 | -131.68 |
| hot_rank_baseline | hot_rank_patched | -0.0688 | -0.1921 | -0.0012 | -39.10 | -194.71 |
