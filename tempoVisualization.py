import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('SpotifyFeatures.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
genre_tempo = df.groupby('genre')['tempo'].mean().sort_values(ascending=False)
genre_tempo.plot(kind='bar', color='skyblue', ax=ax1)
ax1.set_title('Average Tempo by Genre')
ax1.set_xlabel('Genre')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=8)


genre_popularity = df.groupby('genre')['popularity'].mean().sort_values(ascending=False)
genre_popularity.plot(kind='bar', color='salmon', ax=ax2)
ax2.set_title('Average Popularity by Genre')
ax2.set_xlabel('Genre')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=8)


plt.tight_layout()
plt.show()