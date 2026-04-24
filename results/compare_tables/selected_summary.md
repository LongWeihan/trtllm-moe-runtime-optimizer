# Collected Metrics

| Label | Requests | Batches | Avg Req/Batch | Avg Batch ms | Step Std ms | TTFT p90 s | E2E p90 s | TPOT p90 s | Throughput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_moe_default | 12 | 3 | 4.00 | 1597.79 | 233.71 | 0.0809 | 1.4786 | 0.0115 | 280.39 |
| hot_expert_default | 12 | 3 | 4.00 | 1911.58 | 202.50 | 0.0740 | 1.8421 | 0.0114 | 301.32 |
| hot_rank_default | 12 | 3 | 4.00 | 2068.25 | 203.79 | 0.0803 | 1.9107 | 0.0112 | 293.97 |
| mixed_burst_default | 12 | 3 | 4.00 | 1824.93 | 296.38 | 0.0878 | 1.9723 | 0.0141 | 263.02 |
| repeated_prefix_moe_default | 12 | 3 | 4.00 | 1848.56 | 195.04 | 0.0862 | 1.7533 | 0.0140 | 242.35 |
| balanced_moe_max_utilization | 12 | 3 | 4.00 | 1584.58 | 178.96 | 0.0737 | 1.4768 | 0.0115 | 282.72 |
| hot_expert_max_utilization | 12 | 3 | 4.00 | 1918.68 | 216.72 | 0.0734 | 1.8643 | 0.0116 | 300.21 |
| hot_rank_max_utilization | 12 | 3 | 4.00 | 2071.55 | 196.13 | 0.0738 | 1.9273 | 0.0113 | 293.50 |
| mixed_burst_max_utilization | 12 | 3 | 4.00 | 1831.05 | 299.35 | 0.0725 | 1.9917 | 0.0142 | 262.14 |
| repeated_prefix_moe_max_utilization | 12 | 3 | 4.00 | 1856.47 | 182.52 | 0.0725 | 1.7469 | 0.0141 | 241.32 |
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
| hot_expert_moe_v2_replay | 12 | 6 | 2.00 | 1697.70 | 181.13 | 0.0660 | 1.7928 | 0.0112 | 169.64 |
| hot_rank_moe_v2_replay | 12 | 12 | 1.00 | 1531.37 | 164.69 | 0.0115 | 1.7186 | 0.0100 | 99.26 |
| mixed_burst_moe_v2_replay | 12 | 5 | 2.40 | 1561.32 | 168.58 | 0.0758 | 1.5660 | 0.0115 | 184.46 |
| repeated_prefix_moe_moe_v2_replay | 12 | 8 | 1.50 | 1283.00 | 109.51 | 0.0721 | 1.2848 | 0.0115 | 130.94 |
| hot_expert_baseline | 12 | 3 | 4.00 | 1911.58 | 202.50 | 0.0740 | 1.8421 | 0.0114 | 301.32 |
| hot_expert_patched | 12 | 6 | 2.00 | 1697.70 | 181.13 | 0.0660 | 1.7928 | 0.0112 | 169.64 |
| hot_rank_baseline | 12 | 3 | 4.00 | 2068.25 | 203.79 | 0.0803 | 1.9107 | 0.0112 | 293.97 |
| hot_rank_patched | 12 | 12 | 1.00 | 1531.37 | 164.69 | 0.0115 | 1.7186 | 0.0100 | 99.26 |

## Comparisons

| Baseline | Candidate | TTFT p90 delta s | E2E p90 delta s | TPOT p90 delta s | Step Std delta ms | Throughput delta tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| balanced_moe_default | balanced_moe_max_utilization | -0.0072 | -0.0018 | 0.0001 | -54.75 | 2.34 |
| balanced_moe_default | balanced_moe_moe_v1_replay | -0.0036 | 0.0075 | 0.0001 | -39.91 | 0.23 |
| balanced_moe_default | balanced_moe_moe_v2_replay | -0.0013 | -0.0245 | 0.0002 | -93.49 | 25.40 |
| hot_expert_default | hot_expert_max_utilization | -0.0006 | 0.0222 | 0.0001 | 14.21 | -1.12 |
| hot_expert_default | hot_expert_moe_v1_replay | -0.0633 | -0.2723 | -0.0016 | 0.97 | -202.89 |
| hot_expert_default | hot_expert_moe_v2_replay | -0.0080 | -0.0493 | -0.0002 | -21.37 | -131.68 |
| hot_rank_default | hot_rank_max_utilization | -0.0065 | 0.0167 | 0.0001 | -7.66 | -0.47 |
| hot_rank_default | hot_rank_moe_v1_replay | -0.0689 | -0.1983 | -0.0014 | -37.95 | -193.90 |
| hot_rank_default | hot_rank_moe_v2_replay | -0.0688 | -0.1921 | -0.0012 | -39.10 | -194.71 |
| mixed_burst_default | mixed_burst_max_utilization | -0.0152 | 0.0194 | 0.0001 | 2.97 | -0.88 |
| mixed_burst_default | mixed_burst_moe_v1_replay | -0.0141 | -0.3595 | -0.0026 | -125.98 | -84.74 |
| mixed_burst_default | mixed_burst_moe_v2_replay | -0.0120 | -0.4063 | -0.0026 | -127.80 | -78.57 |
| repeated_prefix_moe_default | repeated_prefix_moe_max_utilization | -0.0138 | -0.0064 | 0.0000 | -12.51 | -1.03 |
| repeated_prefix_moe_default | repeated_prefix_moe_moe_v1_replay | -0.0164 | -0.3406 | -0.0027 | -6.88 | -114.07 |
| repeated_prefix_moe_default | repeated_prefix_moe_moe_v2_replay | -0.0141 | -0.4685 | -0.0025 | -85.53 | -111.41 |
| hot_expert_baseline | hot_expert_patched | -0.0080 | -0.0493 | -0.0002 | -21.37 | -131.68 |
| hot_rank_baseline | hot_rank_patched | -0.0688 | -0.1921 | -0.0012 | -39.10 | -194.71 |
