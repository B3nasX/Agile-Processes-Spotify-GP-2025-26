import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importDataset

from collections import Counter


# Load data
df = importDataset.loadData()

# 1. RADAR CHART FOR RANDOM SONG
print("=== RADAR CHART ===")
# Select a random song
random_song = df.sample(1).iloc[0]

# Define features for radar chart
features = ['danceability', 'energy', 'speechiness', 'acousticness', 
        'instrumentalness', 'liveness', 'valence', 'tempo', 'loudness']

# Get actual min/max values from your dataset for proper scaling
feature_ranges = {
    'danceability': (0, 1),
    'energy': (0, 1),
    'speechiness': (0, 1),
    'acousticness': (0, 1),
    'instrumentalness': (0, 1),
    'liveness': (0, 1),
    'valence': (0, 1),
    'tempo': (df['tempo'].min(), df['tempo'].max()),
    'loudness': (df['loudness'].min(), df['loudness'].max())
}

# Normalize values to 0-1 scale based on actual ranges
values = []
for feature in features:
    min_val, max_val = feature_ranges[feature]
    raw_value = random_song[feature]
    # Handle cases where min equals max to avoid division by zero
    if max_val == min_val:
        normalized_value = 0.5
    else:
        normalized_value = (raw_value - min_val) / (max_val - min_val)
        # Ensure value stays within [0,1] range
        normalized_value = max(0, min(1, normalized_value))
    values.append(normalized_value)

# Number of variables
categories = features
N = len(categories)

# Create the radar chart
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Compute angles for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Add the first value at the end to close the polygon
values += values[:1]

# Plot the radar chart
ax.plot(angles, values, 'o-', linewidth=2, label=f"{random_song['track_name']}", color='purple')
ax.fill(angles, values, alpha=0.25, color='purple')

# Add category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# Set y-axis limits
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

# Add grid
ax.grid(True)

# Add title
plt.title(f"Audio Features Radar Chart\n\"{random_song['track_name']}\" by {random_song['artist_name']}\nGenre: {random_song['genre']}", 
        size=14, fontweight='bold', pad=20)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Display song information with actual values
print(f"Random Song Selected:")
print(f"Track: {random_song['track_name']}")
print(f"Artist: {random_song['artist_name']}")
print(f"Genre: {random_song['genre']}")
print(f"Popularity: {random_song['popularity']}")
print("\nAudio Features (Actual Values):")
for feature in features:
    print(f"{feature}: {random_song[feature]:.3f}")

print(f"\nFeature Ranges Used for Scaling:")
for feature in features:
    min_val, max_val = feature_ranges[feature]
    print(f"{feature}: {min_val:.3f} to {max_val:.3f}")

plt.tight_layout()
plt.show()

# 2. KEY DISTRIBUTION BAR CHART
print("\n=== KEY DISTRIBUTION ===")
# Separate data by mode (major/minor)
major_data = df[df['mode'] == 'Major']
minor_data = df[df['mode'] == 'Minor']

# Get all unique keys and sort them in a logical order
key_order = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Filter to only include keys that exist in our data
existing_keys = df['key'].unique()
key_order = [key for key in key_order if key in existing_keys]

# Prepare data for plotting
major_counts = []
minor_counts = []
key_labels = []

for key in key_order:
    major_count = len(major_data[major_data['key'] == key])
    minor_count = len(minor_data[minor_data['key'] == key])
    major_counts.append(major_count)
    minor_counts.append(minor_count)
    key_labels.append(key)

# Create positions for the bars
x_pos = np.arange(len(key_order))
bar_width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Create bars for major and minor
bars1 = ax.bar(x_pos - bar_width/2, major_counts, bar_width, label='Major', color='skyblue', alpha=0.8)
bars2 = ax.bar(x_pos + bar_width/2, minor_counts, bar_width, label='Minor', color='salmon', alpha=0.8)

# Customize the plot
ax.set_xlabel('Key')
ax.set_ylabel('Number of Songs')
ax.set_title('Distribution of Songs by Key and Mode')
ax.set_xticks(x_pos)
ax.set_xticklabels(key_labels)
ax.legend()

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only add label if height > 0
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)

add_value_labels(bars1)
add_value_labels(bars2)

# Calculate and display some statistics
total_songs = len(df)
print(f"Total songs: {total_songs}")
print(f"\nKey Distribution:")
for i, key in enumerate(key_order):
    total_key = major_counts[i] + minor_counts[i]
    if total_key > 0:
        major_percent = (major_counts[i] / total_key) * 100
        minor_percent = (minor_counts[i] / total_key) * 100
        print(f"{key}: {total_key} songs ({major_percent:.1f}% major, {minor_percent:.1f}% minor)")

# Most common key overall
most_common_key = df['key'].mode()[0]
print(f"\nMost common key overall: {most_common_key}")

# Most common key for major mode
most_common_major = major_data['key'].mode()[0] if len(major_data) > 0 else "N/A"
print(f"Most common key for major mode: {most_common_major}")

# Most common key for minor mode
most_common_minor = minor_data['key'].mode()[0] if len(minor_data) > 0 else "N/A"
print(f"Most common key for minor mode: {most_common_minor}")

plt.tight_layout()
plt.show()

# PIE CHART - Genre Distribution (Top 10 genres)
print("\n=== PIE CHART ===")
top_genres = df['genre'].value_counts().head(10)
plt.figure(figsize=(10, 8))
plt.pie(top_genres.values, labels=top_genres.index, autopct='%1.1f%%', startangle=90)
plt.title('Top 10 Genre Distribution')
plt.axis('equal')
plt.tight_layout()
plt.show()

print("Top 10 Genres:")
for genre, count in top_genres.items():
    print(f"{genre}: {count} songs")

