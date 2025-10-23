import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from typing import List

VALID_RANGES = {
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
    'valence': (0, 1),
}


def plot_boxplots(df: pd.DataFrame, features: List[str]) -> None: # Plot boxplots for given features
    if not features:
        print('No features found to plot.')
        return

    n = len(features) # number of features
    cols = min(3, n) # max 3 columns
    rows = math.ceil(n / cols) # calculate required rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3)) # create subplots
    axes = axes.ravel() if hasattr(axes, 'ravel') else [axes] # flatten axes array

    for i, feat in enumerate(features): # iterate over features
        data = df[feat].dropna() # drop NaN values
        bp = axes[i].boxplot(data, vert=True, patch_artist=True) # create boxplot
        # Color (outliers) red for visibility
        for flier in bp.get('fliers', []): # color fliers red
            flier.set(marker='o', color='red', alpha=0.6) # set flier properties

        # draw valid range if available
        lo, hi = VALID_RANGES.get(feat, (None, None)) # get valid range
        if lo is not None: # draw lower bound
            axes[i].axhline(lo, color='green', linestyle='--', linewidth=1) # draw line
        if hi is not None: # draw upper bound
            axes[i].axhline(hi, color='green', linestyle='--', linewidth=1) # draw line

        axes[i].set_title(feat) # set title
        axes[i].set_ylabel('Value') # set y-label

    # hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout() # adjust layout
    plt.suptitle('Boxplots (only)', y=1.02) # set super title
    plt.show()# display plot


def main():
    df = pd.read_csv('SpotifyFeatures.csv')
    features = [c for c in VALID_RANGES.keys() if c in df.columns]
    plot_boxplots(df, features)


if __name__ == '__main__':
    main()