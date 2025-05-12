import utils.session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, lognorm, ks_2samp, probplot, anderson, chisquare, expon, weibull_min, skew, kurtosis, normaltest
from scipy.special import gammainc
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

# Load data (use CSV if possible - faster than Excel)
df = pd.read_excel("mts.xlsx", sheet_name="Sheet1", engine='openpyxl')  # Specify engine
df = df.dropna(subset=["MTN", "GLO", "Airtel", "Time"])

# Convert to float
mtn_speeds = df["MTN"].astype(float)
glo_speeds = df["GLO"].astype(float)
airtel_speeds = df["Airtel"].astype(float)

# # Plot ALL histograms in a single figure (faster)
# plt.figure(figsize=(15, 5))  # Wider figure for 3 subplots
# print("here")

# # MTN
# plt.subplot(1, 3, 1)  # 1 row, 3 columns, position 1
# plt.hist(mtn_speeds, bins=20, density=True, alpha=0.6, color='blue')
# plt.xlabel('Speed (Mbps)')
# plt.ylabel('Density')
# plt.title('MTN Speed Distribution')

# # GLO
# plt.subplot(1, 3, 2)  # Position 2
# plt.hist(glo_speeds, bins=15, density=True, alpha=0.6, color='green')
# plt.xlabel('Speed (Mbps)')
# plt.title('GLO Speed Distribution')

# # Airtel
# plt.subplot(1, 3, 3)  # Position 3
# plt.hist(airtel_speeds, bins=20, density=True, alpha=0.6, color='orange')
# plt.xlabel('Speed (Mbps)')
# plt.title('Airtel Speed Distribution')

# plt.tight_layout()  # Prevent overlapping labels
# plt.show()  # Render once

# # Estimate Gamma parameters (k=shape, theta=scale)
k, loc, theta = gamma.fit(glo_speeds, floc=0)  # Force location=0 for positive speeds
# print(f"Gamma Parameters: k={k:.2f}, θ={theta:.2f}")

# # Generate PDF values
x = np.linspace(0, glo_speeds.max(), 100)
# pdf_gamma = gamma.pdf(x, k, scale=theta)

# # Plot histogram + Gamma PDF
# plt.hist(glo_speeds, bins=15, density=True, alpha=0.6, color='green', label='GLO Data')
# plt.plot(x, pdf_gamma, 'r-', lw=2, label=f'Gamma PDF (k={k:.2f}, θ={theta:.2f})')
# plt.legend()
# plt.show()

s_mtn, loc_mtn, scale_mtn = lognorm.fit(mtn_speeds, floc=0)  # Returns 3 parameters!
mu_mtn = np.log(scale_mtn)  # Convert scale to μ (mean of log(X))
sigma_mtn = s_mtn           # s is the σ (std dev of log(X))
print(f"MTN Log-Normal: μ={mu_mtn:.2f}, σ={sigma_mtn:.2f}")

# Fit log-normal to Airtel speeds
s_airtel, loc_airtel, scale_airtel = lognorm.fit(airtel_speeds, floc=0)
mu_airtel = np.log(scale_airtel)
sigma_airtel = s_airtel
print(f"Airtel Log-Normal: μ={mu_airtel:.2f}, σ={sigma_airtel:.2f}")

# Generate x-values for PDF plots
x = np.linspace(0, max(mtn_speeds.max(), airtel_speeds.max()), 100)

# Plot MTN
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(mtn_speeds, bins=20, density=True, alpha=0.6, color='blue', label='MTN Data')
pdf_mtn = lognorm.pdf(x, s_mtn, scale=scale_mtn)
plt.plot(x, pdf_mtn, 'k-', lw=2, label=f'Log-Normal (μ={mu_mtn:.2f}, σ={sigma_mtn:.2f})')
plt.xlabel('Speed (Mbps)')
plt.ylabel('Density')
plt.title('MTN Speed Distribution')
plt.legend()

# Plot Airtel
plt.subplot(1, 2, 2)
plt.hist(airtel_speeds, bins=20, density=True, alpha=0.6, color='orange', label='Airtel Data')
pdf_airtel = lognorm.pdf(x, s_airtel, scale=scale_airtel)
plt.plot(x, pdf_airtel, 'k-', lw=2, label=f'Log-Normal (μ={mu_airtel:.2f}, σ={sigma_airtel:.2f})')
plt.xlabel('Speed (Mbps)')
plt.title('Airtel Speed Distribution')
plt.legend()

plt.tight_layout()
plt.show()


ks_stat, ks_p = ks_2samp(glo_speeds, gamma.rvs(k, scale=theta, size=1000))
print(f"KS Test for Gamma: Statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
# Interpretation: p-value > 0.05 → Good fit

ks_stat, ks_p = ks_2samp(mtn_speeds, lognorm.rvs(sigma_mtn, scale=np.exp(mu_mtn), size=1000))
print(f"KS Test mtn for Log-Normal: Statistic={ks_stat:.3f}, p-value={ks_p:.3f}")

ks_stat, ks_p = ks_2samp(airtel_speeds, lognorm.rvs(sigma_airtel, scale=np.exp(mu_airtel), size=1000))
print(f"KS Test airtel for Log-Normal: Statistic={ks_stat:.3f}, p-value={ks_p:.3f}")


probplot(glo_speeds, dist=gamma(k, scale=theta), plot=plt)
plt.title('Q-Q Plot for Gamma Fit (GLO)')
plt.show()

# Q-Q Plot for Log-Normal (MTN)
probplot(mtn_speeds, dist=lognorm(sigma_mtn, scale=np.exp(mu_mtn)), plot=plt)
plt.title('Q-Q Plot for Log-Normal Fit (MTN)')
plt.show()

# Q-Q Plot for Log-Normal (Airtel)
probplot(airtel_speeds, dist=lognorm(sigma_airtel, scale=np.exp(mu_airtel)), plot=plt)
plt.title('Q-Q Plot for Log-Normal Fit (Airtel)')
plt.show()

# 1. KS Test
ks_stat, ks_p = ks_2samp(glo_speeds, gamma.rvs(k, scale=theta, size=1000))
print(f"\nKS Test for Gamma (GLO):")
print(f"Statistic={ks_stat:.3f}, p-value={ks_p:.3f}")

# 2. Custom Anderson-Darling Test for Gamma
def anderson_gamma(data, k, theta):
    n = len(data)
    sorted_data = np.sort(data)
    cdf = gamma.cdf(sorted_data, k, scale=theta)

    sum_term = 0
    for i in range(n):
        term1 = (2*i + 1) * np.log(cdf[i])
        term2 = (2*(n-i) - 1) * np.log(1 - cdf[i])
        sum_term += term1 + term2

    A2 = -n - (1/n) * sum_term
    critical_values = [0.576, 0.656, 0.787, 0.918, 1.092]
    significance_level = [10, 5, 2.5, 1, 0.5]
    return A2, critical_values, significance_level

A2, critical_values, sig_level = anderson_gamma(glo_speeds, k, theta)
print(f"\nCustom Anderson-Darling Test for Gamma (GLO):")
print(f"A² statistic: {A2:.3f}")
print(f"Critical values: {critical_values}")
print(f"At 5% significance: {'Good fit' if A2 < critical_values[1] else 'Poor fit'}")

# 3. Robust Chi-Square Test Implementation
def chi_square_gamma_test(data, k, theta, min_bins=5, min_expected=5):
    # Create initial bins (using square root rule)
    n_bins = max(min_bins, int(np.sqrt(len(data))))
    observed, bin_edges = np.histogram(data, bins=n_bins)

    # Calculate expected frequencies
    expected = []
    for i in range(len(bin_edges)-1):
        prob = gamma.cdf(bin_edges[i+1], k, scale=theta) - gamma.cdf(bin_edges[i], k, scale=theta)
        expected.append(prob * len(data))
    expected = np.array(expected)

    # Combine bins to meet minimum expected frequency
    while True:
        # Check if all expected >= min_expected
        if np.all(expected >= min_expected) or len(expected) <= 2:
            break

        # Find the bin with smallest expected frequency
        min_idx = np.argmin(expected)

        # Merge with adjacent bin (prefer merging with right neighbor for left-most bins)
        if min_idx == 0 or (min_idx < len(expected)-1 and expected[min_idx+1] < expected[min_idx-1]):
            # Merge with right neighbor
            expected[min_idx+1] += expected[min_idx]
            observed[min_idx+1] += observed[min_idx]
        else:
            # Merge with left neighbor
            expected[min_idx-1] += expected[min_idx]
            observed[min_idx-1] += observed[min_idx]

        # Remove the merged bin
        expected = np.delete(expected, min_idx)
        observed = np.delete(observed, min_idx)

    # Normalize expected frequencies to match observed sum
    expected = expected * np.sum(observed) / np.sum(expected)

    return chisquare(observed, expected, ddof=2)  # ddof=2 for estimated parameters (k and theta)

chi2_stat, p_value = chi_square_gamma_test(glo_speeds, k, theta)
print(f"\nChi-Square Test for Gamma (GLO):")
print(f"χ² statistic: {chi2_stat:.3f}")
print(f"p-value: {p_value:.3f}")
print("Interpretation:", "Good fit" if p_value > 0.05 else "Poor fit")


# Try Exponential distribution
loc_exp, scale_exp = expon.fit(glo_speeds)
print(f"Exponential λ = {1/scale_exp:.3f}")

# Try Weibull distribution
c, loc_weib, scale_weib = weibull_min.fit(glo_speeds, floc=0)
print(f"Weibull c = {c:.3f}, scale = {scale_weib:.3f}")

# Probability of speed < 1 Mbps
prob_slow_weib = weibull_min.cdf(1, 1.159, scale=0.701)
print(f"Probability speed < 1 Mbps: {prob_slow_weib:.1%}")

# 95th percentile speed
speed_95 = weibull_min.ppf(0.95, 1.159, scale=0.701)
print(f"95th percentile speed: {speed_95:.2f} Mbps")


# def weibull_performance_metrics(c, scale):
#     metrics = {
#         'mean_speed': scale * np.exp(np.log(gamma(1 + 1/c))),
#         'median_speed': scale * (np.log(2))**(1/c),
#         'speed_variation': (scale**2) * (gamma(1 + 2/c) - (gamma(1 + 1/c))**2)
#     }
#     return metrics

# print(weibull_performance_metrics(1.159, 0.701))

# Convert time objects to strings first
df['Time_str'] = df['Time'].apply(lambda x: x.strftime('%H:%M:%S'))

# Now convert to datetime
df['Time_dt'] = pd.to_datetime(df['Time_str'], format='%H:%M:%S')

# Create 12-hour formatted column
df['Time_12hr'] = df['Time_dt'].dt.strftime('%I:%M %p')

# Sort by actual time (not string)
df = df.sort_values('Time_dt')

# Create time-series plot
plt.figure(figsize=(14,6))
for provider, color in zip(['MTN', 'GLO', 'Airtel'], ['#1f77b4', '#2ca02c', '#d62728']):
    plt.plot(df['Time_12hr'], df[provider],
             color=color, label=provider, marker='o', linestyle='-', linewidth=1, markersize=5)

plt.xticks(rotation=45)
plt.xlabel('Time (12-hour)')
plt.ylabel('Speed (Mbps)')
plt.title('Network Speed Measurements (4:00 AM - 6:00 AM)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()


stats = pd.DataFrame({
    'MTN': {
        'Mean': df['MTN'].mean(),
        'Median': df['MTN'].median(),
        'Std Dev': df['MTN'].std(),
        '25th %ile': df['MTN'].quantile(0.25),
        '75th %ile': df['MTN'].quantile(0.75),
        'Skewness': skew(df['MTN'].dropna()),
        'Kurtosis': kurtosis(df['MTN'].dropna())
    },
    'GLO': {
        'Mean': df['GLO'].mean(),
        'Median': df['GLO'].median(),
        'Std Dev': df['GLO'].std(),
        '25th %ile': df['GLO'].quantile(0.25),
        '75th %ile': df['GLO'].quantile(0.75),
        'Skewness': skew(df['GLO'].dropna()),
        'Kurtosis': kurtosis(df['GLO'].dropna())
    },
    'Airtel': {
        'Mean': df['Airtel'].mean(),
        'Median': df['Airtel'].median(),
        'Std Dev': df['Airtel'].std(),
        '25th %ile': df['Airtel'].quantile(0.25),
        '75th %ile': df['Airtel'].quantile(0.75),
        'Skewness': skew(df['Airtel'].dropna()),
        'Kurtosis': kurtosis(df['Airtel'].dropna())
    }
}).round(2)

print(stats)

for provider in ['MTN', 'GLO', 'Airtel']:
    stat, p = normaltest(df[provider].dropna())
    print(f"{provider}: p-value = {p:.4f} ({'Not normal' if p < 0.05 else 'Normal'})")


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['MTN', 'GLO', 'Airtel']], palette=['#1f77b4', '#2ca02c', '#d62728'])
plt.title('Internet Speed Distribution by Provider (PM Hours)', fontsize=14)
plt.ylabel('Speed (Mbps)', fontsize=12)
plt.xticks(fontsize=12)

# Annotate outliers
for i, provider in enumerate(['MTN', 'GLO', 'Airtel']):
    q1 = df[provider].quantile(0.25)
    q3 = df[provider].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[provider] > q3 + 1.5*iqr) | (df[provider] < q1 - 1.5*iqr)][provider]
    for val in outliers:
        plt.text(i, val, f'{val:.1f}', ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

hourly_stats = df.groupby(df['Time_dt'].dt.hour).agg({
    'MTN': ['mean', 'std', 'count'],
    'GLO': ['median', lambda x: (x < 0.5).mean()],
    'Airtel': ['max', 'min']
}).rename(index={16: '4 PM', 17: '5 PM', 18: '6 PM'})

print(hourly_stats.round(2))


def detect_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series[(series > q3 + 1.5*iqr) | (series < q1 - 1.5*iqr)]

outliers = {
    'MTN': detect_outliers(df['MTN']),
    'GLO': detect_outliers(df['GLO']),
    'Airtel': detect_outliers(df['Airtel'])
}

for provider, vals in outliers.items():
    print(f"{provider} outliers ({len(vals)}): {', '.join([f'{v:.2f}' for v in vals])}")


heatmap_data = df.pivot_table(
    index='Date',
    columns=df['Time_dt'].dt.hour,
    values='MTN',
    aggfunc='mean'
)

plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f")
plt.title('MTN Speeds by Date and Hour')
plt.xlabel('Hour (24h)')
plt.ylabel('Date')
plt.show()