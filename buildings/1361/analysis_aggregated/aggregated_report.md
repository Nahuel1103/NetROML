# Aggregated RSSI Analysis Report
## Pair Identification
- **Client**: 132.56.56.164.101.173
- **AP**: 136.29.252.141.130.32
- **Months Included**: 03, 05, 07, 09, 11

## Statistics (Aggregated)
- **Sample Size (N)**: 836
- **Mean RSSI**: -74.6244 dBm
- **Median RSSI**: -74.5000 dBm
- **Variance**: 66.5965
- **Standard Deviation**: 8.1607
- **95% Confidence Interval**: [-75.1784, -74.0704]

## Hypothesis Testing
### Student's T-test (One-sample)
- **Null Hypothesis ($H_0$)**: The population mean is -75.0 dBm.
- **Statistic**: t = 1.3308
- **p-value**: 1.8363e-01
- **Result**: Fail to reject $H_0$ (at 5% significance level).

### Chi-squared Normality Test
- **Null Hypothesis ($H_0$)**: The distribution is Normal.
- **Statistic**: $\chi^2$ = 5.4967
- **p-value**: 6.4033e-02
- **Result**: Fail to reject $H_0$ (Likely Normal).

## Findings
The distribution of RSSI values shows a mean of -74.6244 dBm. The T-test indicates that this is not significantly different from -75.0 dBm. Additionally, the Chi-squared normality test suggests the data is potentially normally distributed.
