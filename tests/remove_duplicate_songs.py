import pandas as pd
import importDataset

"""
This script removes duplicate songs from the Spotify dataset.
Duplicates are defined as songs with the same track_name and artist_name 
but different genre tags.

The script keeps only the first occurrence of each unique (track_name, artist_name) pair.
"""

def remove_duplicates():
    """Remove duplicate songs and save cleaned dataset"""
    
    # Load the original dataset
    print("Loading dataset...")
    df = importDataset.loadData()
    
    print(f"Original dataset size: {len(df)} rows")
    
    # Show some examples of duplicates before removal
    print("\n=== EXAMPLES OF DUPLICATES ===")
    duplicates = df[df.duplicated(subset=['track_name', 'artist_name'], keep=False)]
    if not duplicates.empty:
        # Group by track_name and artist_name to show examples
        grouped = duplicates.groupby(['track_name', 'artist_name'])
        
        # Show first 3 examples
        count = 0
        for (track, artist), group in grouped:
            if count >= 3:
                break
            print(f"\nSong: {track}")
            print(f"Artist: {artist}")
            print(f"Number of entries: {len(group)}")
            print("Genres found:")
            for idx, row in group.iterrows():
                print(f"  - {row['genre']}")
            count += 1
        
        total_duplicate_groups = len(grouped)
        print(f"\nTotal songs with duplicates: {total_duplicate_groups}")
        print(f"Total duplicate rows: {len(duplicates)}")
    else:
        print("No duplicates found!")
    
    # Remove duplicates - keep first occurrence
    print("\n=== REMOVING DUPLICATES ===")
    df_cleaned = df.drop_duplicates(subset=['track_name', 'artist_name'], keep='first')
    
    print(f"Cleaned dataset size: {len(df_cleaned)} rows")
    print(f"Rows removed: {len(df) - len(df_cleaned)}")
    
    # Save cleaned dataset
    output_file = "SpotifyFeatures_cleaned.csv"
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_file}")
    
    # Show statistics
    print("\n=== STATISTICS ===")
    print(f"Original unique songs (by track_id): {df['track_id'].nunique()}")
    print(f"Original unique (track_name, artist_name) pairs: {df.groupby(['track_name', 'artist_name']).ngroups}")
    print(f"Cleaned unique songs: {len(df_cleaned)}")
    print(f"Reduction: {((len(df) - len(df_cleaned)) / len(df) * 100):.2f}%")
    
    return df_cleaned


def analyze_duplicates():
    """Analyze duplicate patterns in the dataset"""
    
    print("Loading dataset...")
    df = importDataset.loadData()
    
    print("\n=== DUPLICATE ANALYSIS ===")
    
    # Find all duplicates
    duplicates = df[df.duplicated(subset=['track_name', 'artist_name'], keep=False)]
    
    if duplicates.empty:
        print("No duplicates found!")
        return
    
    # Group by track and artist
    grouped = duplicates.groupby(['track_name', 'artist_name'])
    
    # Count how many genres per song
    genre_counts = grouped['genre'].nunique()
    
    print(f"\nTotal songs with duplicates: {len(grouped)}")
    print(f"Total duplicate rows: {len(duplicates)}")
    print(f"\nGenre distribution for duplicates:")
    print(f"  Songs with 2 genres: {(genre_counts == 2).sum()}")
    print(f"  Songs with 3 genres: {(genre_counts == 3).sum()}")
    print(f"  Songs with 4+ genres: {(genre_counts >= 4).sum()}")
    
    # Show songs with most genre variations
    print(f"\n=== TOP 10 SONGS WITH MOST GENRE VARIATIONS ===")
    top_varied = genre_counts.nlargest(10)
    for (track, artist), count in top_varied.items():
        print(f"\n{track} - {artist}")
        print(f"  Number of different genres: {count}")
        genres = grouped.get_group((track, artist))['genre'].unique()
        print(f"  Genres: {', '.join(genres)}")


if __name__ == '__main__':
    print("=" * 60)
    print("SPOTIFY DATASET DUPLICATE REMOVER")
    print("=" * 60)
    
    # First, analyze the duplicates
    analyze_duplicates()
    
    print("\n" + "=" * 60)
    
    # Ask user if they want to proceed
    response = input("\nDo you want to remove duplicates and create a cleaned dataset? (y/n): ")
    
    if response.lower() == 'y':
        df_cleaned = remove_duplicates()
        print("\n✅ Done! You can now use 'SpotifyFeatures_cleaned.csv' in your applications.")
    else:
        print("\nOperation cancelled.")
