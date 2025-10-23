import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('SpotifyFeatures.csv')

# Define the valid ranges for each feature based on your specifications
valid_ranges = {
    'danceability': (0.06, 0.99),
    'duration_ms': (15400, 55500000),
    'energy': (0, 1),
    'instrumentalness': (0, 1),
    'acousticness': (0, 1),
    'popularity': (0, 100),
    'liveness': (0.01, 1),
    'loudness': (-52.5, 3.74),
    'speechiness': (0.02, 0.97),
    'tempo': (30.4, 243),
    'valence': (0, 1)
}


def detect_range_outliers(df, column, valid_range):
    """Detect outliers based on predefined valid ranges"""
    lower_bound, upper_bound = valid_range
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def detect_iqr_outliers(df, column):
    """Detect outliers using IQR method for comparison"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


# Select only numeric columns that have defined ranges
numeric_columns = [col for col in valid_ranges.keys() if col in df.columns]

print("=== OUTLIER ANALYSIS BASED ON DEFINED RANGES ===")
print(f"Dataset shape: {df.shape}")
print(f"Columns analyzed: {len(numeric_columns)}")

# 1. Outlier detection using predefined ranges
outlier_summary = []

for col in numeric_columns:
    if col in df.columns:
        # Range-based outliers
        outliers_range, range_lower, range_upper = detect_range_outliers(df, col, valid_ranges[col])
        # IQR outliers for comparison
        outliers_iqr, iqr_lower, iqr_upper = detect_iqr_outliers(df, col)

        range_outlier_count = len(outliers_range)
        iqr_outlier_count = len(outliers_iqr)
        range_outlier_percentage = (range_outlier_count / len(df)) * 100

        outlier_summary.append({
            'column': col,
            'valid_range': valid_ranges[col],
            'range_outliers': range_outlier_count,
            'iqr_outliers': iqr_outlier_count,
            'range_outlier_percentage': range_outlier_percentage,
            'data_min': df[col].min(),
            'data_max': df[col].max(),
            'data_median': df[col].median()
        })

        print(f"\n{col.upper()}:")
        print(f"  Valid Range: [{range_lower}, {range_upper}]")
        print(f"  Data Range: [{df[col].min():.4f}, {df[col].max():.4f}]")
        print(f"  Range-based Outliers: {range_outlier_count} ({range_outlier_percentage:.2f}%)")
        print(f"  IQR Outliers: {iqr_outlier_count}")

# Create summary dataframe
outlier_df = pd.DataFrame(outlier_summary)

# 2. Visualizations - Range-based outlier detection
fig, axes = plt.subplots(4, 3, figsize=(20, 16))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    if i < len(axes):
        # Create box plot with range boundaries
        bp = axes[i].boxplot(df[col].dropna(), vert=True, patch_artist=True)

        # Color outliers in red
        for flier in bp['fliers']:
            flier.set(marker='o', color='red', alpha=0.5)

        # Add valid range boundaries
        range_lower, range_upper = valid_ranges[col]
        axes[i].axhline(y=range_lower, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Valid Range')
        axes[i].axhline(y=range_upper, color='green', linestyle='--', alpha=0.7, linewidth=2)

        # Add IQR boundaries for comparison
        outliers_iqr, iqr_lower, iqr_upper = detect_iqr_outliers(df, col)
        axes[i].axhline(y=iqr_lower, color='orange', linestyle=':', alpha=0.5, label='IQR Range')
        axes[i].axhline(y=iqr_upper, color='orange', linestyle=':', alpha=0.5)

        axes[i].set_title(f'{col}\nRange Outliers: {outlier_summary[i]["range_outliers"]}', fontsize=10)
        axes[i].set_ylabel('Values')

        if i == 0:
            axes[i].legend()

plt.tight_layout()
plt.suptitle('Outlier Detection - Valid Range vs IQR Method', y=1.02, fontsize=16)
plt.show()

# 3. Violin plots to show distribution with valid ranges
fig, axes = plt.subplots(4, 3, figsize=(20, 16))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    if i < len(axes):
        # Create violin plot
        sns.violinplot(y=df[col], ax=axes[i], color='lightblue')

        # Add valid range boundaries
        range_lower, range_upper = valid_ranges[col]
        axes[i].axhline(y=range_lower, color='red', linestyle='--', linewidth=2, label='Valid Range')
        axes[i].axhline(y=range_upper, color='red', linestyle='--', linewidth=2)

        # Shade the valid range area
        axes[i].axhspan(range_lower, range_upper, alpha=0.2, color='green', label='Valid Zone')

        axes[i].set_title(f'{col}\nValid: [{range_lower}, {range_upper}]', fontsize=10)
        axes[i].set_ylabel('Values')

        if i == 0:
            axes[i].legend()

plt.tight_layout()
plt.suptitle('Distribution Violin Plots with Valid Ranges', y=1.02, fontsize=16)
plt.show()

# 4. Outlier percentage comparison chart
plt.figure(figsize=(12, 8))
bars = plt.bar(outlier_df['column'], outlier_df['range_outlier_percentage'],
               color=['red' if p > 5 else 'orange' if p > 1 else 'green' for p in
                      outlier_df['range_outlier_percentage']])

plt.title('Percentage of Outliers by Column (Based on Valid Ranges)', fontsize=16)
plt.xlabel('Audio Features')
plt.ylabel('Outlier Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, percentage in zip(bars, outlier_df['range_outlier_percentage']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='High Outlier Threshold (5%)')
plt.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Moderate Outlier Threshold (1%)')
plt.legend()
plt.tight_layout()
plt.show()

# 5. Detailed outlier analysis table
print("\n=== DETAILED OUTLIER ANALYSIS ===")
detailed_table = []
for col in numeric_columns:
    outliers, lower, upper = detect_range_outliers(df, col, valid_ranges[col])

    if len(outliers) > 0:
        outlier_stats = outliers[col].describe()
        detailed_table.append({
            'Feature': col,
            'Valid Range': f"[{lower}, {upper}]",
            'Outlier Count': len(outliers),
            'Outlier %': f"{(len(outliers) / len(df)) * 100:.2f}%",
            'Min Outlier': outliers[col].min(),
            'Max Outlier': outliers[col].max(),
            'Avg Outlier': outliers[col].mean()
        })

detailed_df = pd.DataFrame(detailed_table)
print(detailed_df.to_string(index=False))

# 6. Save detailed outlier information
outlier_details = []
for col in numeric_columns:
    outliers, lower, upper = detect_range_outliers(df, col, valid_ranges[col])
    for idx, row in outliers.iterrows():
        outlier_details.append({
            'track_id': row.get('track_id', idx),
            'feature': col,
            'outlier_value': row[col],
            'valid_min': lower,
            'valid_max': upper,
            'deviation': abs(row[col] - upper) if row[col] > upper else abs(row[col] - lower),
            'artist_name': row.get('artist_name', 'N/A'),
            'track_name': row.get('track_name', 'N/A'),
            'genre': row.get('genre', 'N/A')
        })

outlier_details_df = pd.DataFrame(outlier_details)
outlier_details_df.to_csv('range_based_outlier_analysis.csv', index=False)
print(f"\nDetailed outlier analysis saved to 'range_based_outlier_analysis.csv'")

# 7. Summary and recommendations
print("\n=== CLEANING RECOMMENDATIONS ===")
high_outlier_cols = outlier_df[outlier_df['range_outlier_percentage'] > 5]
moderate_outlier_cols = outlier_df[
    (outlier_df['range_outlier_percentage'] > 1) & (outlier_df['range_outlier_percentage'] <= 5)]

if len(high_outlier_cols) > 0:
    print("🚨 HIGH PRIORITY - Consider removing or correcting these outliers:")
    for _, col in high_outlier_cols.iterrows():
        print(f"   {col['column']}: {col['range_outliers']} outliers ({col['range_outlier_percentage']:.1f}%)")

if len(moderate_outlier_cols) > 0:
    print("\n⚠️  MODERATE PRIORITY - Review these outliers:")
    for _, col in moderate_outlier_cols.iterrows():
        print(f"   {col['column']}: {col['range_outliers']} outliers ({col['range_outlier_percentage']:.1f}%)")

clean_cols = outlier_df[outlier_df['range_outlier_percentage'] <= 1]
if len(clean_cols) > 0:
    print("\n✅ CLEAN - These features have minimal outliers:")
    for _, col in clean_cols.iterrows():
        print(f"   {col['column']}: {col['range_outliers']} outliers ({col['range_outlier_percentage']:.1f}%)")

# 8. Data quality summary
total_outliers = sum([summary['range_outliers'] for summary in outlier_summary])
total_possible = len(df) * len(numeric_columns)
data_quality_score = ((total_possible - total_outliers) / total_possible) * 100

print(f"\n=== DATA QUALITY SUMMARY ===")
print(f"Total tracks: {len(df):,}")
print(f"Total outlier instances: {total_outliers:,}")
print(f"Overall data quality score: {data_quality_score:.1f}%")
print(f"Features with outliers: {len([s for s in outlier_summary if s['range_outliers'] > 0])}/{len(numeric_columns)}")