import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import importDataset

from collections import Counter
df = importDataset.loadData()


print("=== RADAR CHART ===")
random_song = df.sample(1).iloc[0]
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

ax.plot(angles, values, 'o-', linewidth=2, label=f"{random_song['track_name']}", color='purple')
ax.fill(angles, values, alpha=0.25, color='purple')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
ax.grid(True)
plt.title(f"Audio Features Radar Chart\n\"{random_song['track_name']}\" by {random_song['artist_name']}\n ({random_song['genre']})")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

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



# PIE CHART 
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

