import pandas as ps
import importDataset
import random
from csv_Py_Scipts import columnVariables

data_set = importDataset.loadData()


def getRandomSong():
    chosen_song = data_set['track_id'].sample(n=1, ignore_index=True)
    song_id = chosen_song
    return chosen_song

def createSpotifyLink(chosen_song):
    song_link = f"https://open.spotify.com/track/{chosen_song.to_string(index=False)}"  
    return song_link

if __name__ == '__main__':
    chosen_song = getRandomSong()
    spotify_link = createSpotifyLink(chosen_song)
    print(spotify_link)
