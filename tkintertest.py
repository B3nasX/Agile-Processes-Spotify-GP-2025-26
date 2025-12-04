import tkinter as tk
import webbrowser
from grabRandomSong import *


root = tk.Tk()

# Preemptive so it actually shows a link immedietly
chosen_song = getRandomSong()
spotify_link = createSpotifyLink(chosen_song)

# This is the function to generate a new song and also change the label
def generateNewSong():
    chosen_song = getRandomSong()
    spotify_link = createSpotifyLink(chosen_song)
    link.config(text=f"{spotify_link}")
    return spotify_link


# Create the "link" with the spotify link
link = tk.Label(root, text=f"{spotify_link}", fg="blue", cursor="hand2")
link.pack()
link.bind("<Button-1>", lambda event: webbrowser.open(link.cget("text")))

# When you click button it runs the generateNewSong function
button = tk.Button(root, text='Generate Song', width=25, command=generateNewSong).pack()
root.mainloop()

