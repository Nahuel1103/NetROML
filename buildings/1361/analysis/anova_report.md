# ANOVA Statistical Analysis Report
## Target: User 132.56.56.164.101.173
- **Objective**: Compare the mean RSSI values across multiple months to detect significant signal variations.
- **Months Analysed**: 03, 05, 07, 09, 11

## Descriptive Statistics
|   Month |   Count |     Mean |   StdDev |   Variance |
|--------:|--------:|---------:|---------:|-----------:|
|      03 |     344 | -86.2849 |  8.9765  |    80.5775 |
|      05 |    2292 | -85.2112 |  9.12175 |    83.2064 |
|      07 |    1448 | -84.6395 | 10.2925  |   105.936  |
|      09 |    1988 | -84.9034 |  9.63442 |    92.8221 |
|      11 |    2512 | -87.0772 |  8.56421 |    73.3457 |

## ANOVA Test Results (One-Way)
- **F-statistic**: 23.9955
- **p-value**: 9.0960e-20
- **Significance Level ($lpha$)**: 0.05

### Conclusion
**There IS a statistically significant difference in mean RSSI across the selected months.**
(Result based on p-value < 0.05).

## Visualization
The boxplot below visually confirms the differences (or lack thereof) in signal distribution patterns between the months.
(Plot saved at: buildings/1361/analysis/plots/anova_boxplot.png)
