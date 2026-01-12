# Aggregated RSSI Analysis Report
## Pair Identification
- **Client**: 32.45.7.240.187.6
- **AP**: 136.29.252.170.206.224
- **Months Included**: 02, 03, 04, 05, 06, 07, 08, 09, 10, 11

## Statistics (Aggregated)
- **Sample Size (N)**: 12
- **Mean RSSI**: -81.6667 dBm
- **Median RSSI**: -87.0000 dBm
- **Variance**: 172.9697
- **Standard Deviation**: 13.1518
- **95% Confidence Interval**: [-90.0229, -73.3104]

## Hypothesis Testing
### Student's T-test (One-sample)
- **Null Hypothesis ($H_0$)**: The population mean is -75.0 dBm.
- **Statistic**: t = -1.7560
- **p-value**: 1.0686e-01
- **Result**: Fail to reject $H_0$ (at 5% significance level).

### Chi-squared Normality Test
- **Null Hypothesis ($H_0$)**: The distribution is Normal.
- **Statistic**: $\chi^2$ = 3.9746
- **p-value**: 1.3706e-01
- **Result**: Fail to reject $H_0$ (Likely Normal).

## Findings
The distribution of RSSI values shows a mean of -81.6667 dBm. The T-test indicates that this is not significantly different from -75.0 dBm. Additionally, the Chi-squared normality test suggests the data is potentially normally distributed.
