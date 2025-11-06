import pandas as ps
import importDataset
from csv_Py_Scipts import columnVariables

data_set = importDataset.loadData()

# One small issue i'd need help debugging -> henri salvador keeps appearing at the top.
# Probably from how the data is imported?
# Maybe he has a null value?

# Also theres a cool optimizations using a switch statement where it could be one function but change depending on user input
# That can be changed later but for now this does the work required and I don't want to make it harder later with tkinter. 
# Cool ideas are a brew

# Maybe could have it output just the song title, album, artist, sorted value? <- seems gimicky and bad :(


def sortByPopularity(data_set):
    print("-Sorted by Popularity-")
    popularity_column = data_set.columns[columnVariables.POPULARITY]
    data_set_popularity = data_set.sort_values(by=popularity_column, ascending=False)
    print(data_set_popularity.head())
    return data_set_popularity

def sortByAcousticness(data_set):
    print("-Sorted by Acousticness-")
    acousticness_column = data_set.columns[columnVariables.ACOUSTICNESS]
    data_set_acousticness = data_set.sort_values(by=acousticness_column, ascending=False)
    print(data_set_acousticness.head())
    return data_set_acousticness

def sortByDanceability(data_set):
    print("-Sorted by Danceability-")
    danceability_column = data_set.columns[columnVariables.DANCEABILITY]
    data_set_danceability = data_set.sort_values(by=danceability_column, ascending=False)
    print(data_set_danceability.head())
    return data_set_danceability

def sortByDuration(data_set):
    print("-Sorted by Duration-")
    duration_column = data_set.columns[columnVariables.DURATION_MS]
    data_set_duration = data_set.sort_values(by=duration_column, ascending=False)
    print(data_set_duration.head())
    return data_set_duration

def sortByEnergy(data_set):
    print("-Sorted by Energy-")
    energy_column = data_set.columns[columnVariables.ENERGY]
    data_set_energy = data_set.sort_values(by=energy_column, ascending=False)
    print(data_set_energy.head())
    return data_set_energy

def sortByInstrumentalness(data_set):
    print("-Sorted by Instrumentalness-")
    instrumentalness_column = data_set.columns[columnVariables.INSTRUMENTALNESS]
    data_set_instrumentalness = data_set.sort_values(by=instrumentalness_column, ascending=False)
    print(data_set_instrumentalness.head())
    return data_set_instrumentalness

def sortByLiveness(data_set):
    print("-Sorted by Liveness-")
    liveness_column = data_set.columns[columnVariables.LIVENESS]
    data_set_liveness = data_set.sort_values(by=liveness_column, ascending=False)
    print(data_set_liveness.head())
    return data_set_liveness

def sortByLoudness(data_set):
    print("-Sorted by Loudness-")
    loudness_column = data_set.columns[columnVariables.LOUDNESS]
    data_set_loudness = data_set.sort_values(by=loudness_column, ascending=False)
    print(data_set_loudness.head())
    return data_set_loudness

def sortBySpeechiness(data_set):
    print("-Sorted by Speechiness-")
    speechiness_column = data_set.columns[columnVariables.SPEECHINESS]
    data_set_speechiness = data_set.sort_values(by=speechiness_column, ascending=False)
    print(data_set_speechiness.head())
    return data_set_speechiness

def sortByTempo(data_set):
    print("-Sorted by Tempo-")
    tempo_column = data_set.columns[columnVariables.TEMPO]
    data_set_tempo = data_set.sort_values(by=tempo_column, ascending=False)
    print(data_set_tempo.head())
    return data_set_tempo

def sortByValence(data_set):
    print("-Sorted by Valence-")
    valence_column = data_set.columns[columnVariables.VALENCE]
    data_set_valence = data_set.sort_values(by=valence_column, ascending=False)
    print(data_set_valence.head())
    return data_set_valence


if __name__ == "__main__":
    print("Pick a column to sort by: ")
    print(
    "1 - Popularity \n" \
    "2 - Acousticness \n" \
    "3 - Danceability \n" \
    "4 - Duration (Miliseconds) \n" \
    "5 - Energy \n" \
    "6 - Instrumentalness \n" \
    "7 - Liveness \n" \
    "8 - Loudness \n" \
    "9 - Speechiness \n" \
    "10 - Tempo \n" \
    "11 - Valence")

    user_choice = input("=> ")

    match user_choice:
        case "1":
            sorted_data_set = sortByPopularity(data_set) 
        case "2":
            sorted_data_set = sortByAcousticness(data_set) 
        case "3":
            sorted_data_set = sortByDanceability(data_set) 
        case "4":
            sorted_data_set = sortByDuration(data_set) 
        case "5":
            sorted_data_set = sortByEnergy(data_set) 
        case "6":
            sorted_data_set = sortByInstrumentalness(data_set) 
        case "7":
            sorted_data_set = sortByLiveness(data_set) 
        case "8":
            sorted_data_set = sortByLoudness(data_set) 
        case "9":
            sorted_data_set = sortBySpeechiness(data_set) 
        case "10":
            sorted_data_set = sortByTempo(data_set) 
        case "11":
            sorted_data_set = sortByValence(data_set) 
        case _:
            print("Invalid input")
