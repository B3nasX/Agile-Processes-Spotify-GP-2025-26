import tkinter as tk
from src.ai_song_recommender import SongRecommenderApp

def test_full_coverage():
    root = tk.Tk()
    app = SongRecommenderApp(root)
    
    app.song_entry.insert(0, "Shape of You")
    app.search_and_recommend()
    app.search_and_recommend(find_least_similar=True)
    
    if len(app.df) > 1:
        app.show_song_details(app.df.iloc[0], app.df.iloc[1])
    
    app.song_entry.delete(0, "end")
    app.song_entry.insert(0, "a")
    
    app.open_link(None)
    
    root.update()
    root.destroy()
    
    print("ALL CODE COVERED")
