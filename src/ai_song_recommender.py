import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import importDataset


class SongRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Song Recommender")
        self.root.geometry("1000x750")
        self.root.configure(bg="#1a1a1a")
        
        # Load dataset
        self.load_data()
        
        # Create GUI
        self.create_widgets()
        
    def load_data(self):
        """Load and prepare the Spotify dataset"""
        print("Loading dataset...")
        self.df = importDataset.loadData()
        
        # Audio features to use for similarity calculation
        self.feature_columns = [
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
        ]
        
        # Prepare feature matrix
        self.prepare_features()
        print("Dataset loaded successfully!")
        
    def prepare_features(self):
        """Normalize audio features for similarity calculation"""
        # Extract features and handle missing values
        self.features_df = self.df[self.feature_columns].fillna(0)

        # Normalize features using StandardScaler
        # Fit on numpy array to avoid future warnings about feature names
        self.scaler = StandardScaler()
        self.normalized_features = self.scaler.fit_transform(self.features_df.values)
        
    def create_widgets(self):
        """Create the GUI components"""
        # Title
        title_label = tk.Label(
            self.root,
            text="AI Song Recommender",
            font=("Helvetica", 28, "bold"),
            bg="#1a1a1a",
            fg="#1DB954"
        )
        title_label.pack(pady=30)
        
        # Search Frame
        search_frame = tk.Frame(self.root, bg="#1a1a1a")
        search_frame.pack(pady=15, padx=30, fill="x")
        
        # Row 1: Search Label and Entry
        row1_frame = tk.Frame(search_frame, bg="#1a1a1a")
        row1_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(
            row1_frame,
            text="Search:",
            font=("Helvetica", 13),
            bg="#1a1a1a",
            fg="#ffffff"
        ).pack(side="left", padx=5)
        
        # Entry with autocomplete container
        entry_container = tk.Frame(row1_frame, bg="#1a1a1a")
        entry_container.pack(side="left", padx=5)
        
        self.song_entry = tk.Entry(
            entry_container,
            font=("Helvetica", 12),
            width=45,
            bg="#2a2a2a",
            fg="#ffffff",
            insertbackground="#ffffff",
            relief=tk.FLAT,
            highlightthickness=2,
            highlightbackground="#3a3a3a",
            highlightcolor="#1DB954"
        )
        self.song_entry.pack()
        self.song_entry.bind("<KeyRelease>", self.on_key_release)
        self.song_entry.bind("<Return>", lambda e: self.search_and_recommend())
        self.song_entry.bind("<Down>", self.on_arrow_down)
        
        # Autocomplete listbox (hidden by default)
        self.autocomplete_frame = tk.Frame(entry_container, bg="#2a2a2a", relief=tk.FLAT, borderwidth=0)
        self.autocomplete_listbox = tk.Listbox(
            self.autocomplete_frame,
            font=("Helvetica", 10),
            height=8,
            width=60,
            selectmode=tk.SINGLE,
            bg="#2a2a2a",
            fg="#ffffff",
            selectbackground="#1DB954",
            selectforeground="#000000",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor="#1DB954"
        )
        self.autocomplete_listbox.pack(fill="both", expand=True)
        self.autocomplete_listbox.bind("<ButtonRelease-1>", self.on_listbox_click)
        self.autocomplete_listbox.bind("<Return>", self.on_select_suggestion)
        
        # Store song data for autocomplete
        self.suggestion_data = []
        
        # Row 2: Buttons and Options
        row2_frame = tk.Frame(search_frame, bg="#1a1a1a")
        row2_frame.pack(fill="x")
        
        # Buttons Frame
        btn_frame = tk.Frame(row2_frame, bg="#1a1a1a")
        btn_frame.pack(side="left")
        
        search_btn = tk.Button(
            btn_frame,
            text="Find Similar Songs",
            font=("Helvetica", 11, "bold"),
            bg="#1DB954",
            fg="#000000",
            command=self.search_and_recommend,
            cursor="hand2",
            padx=20,
            pady=8,
            relief=tk.FLAT,
            activebackground="#1ed760",
            activeforeground="#000000"
        )
        search_btn.pack(side="left", padx=5)
        
        least_similar_btn = tk.Button(
            btn_frame,
            text="Find Least Similar",
            font=("Helvetica", 11, "bold"),
            bg="#e91e63",
            fg="#ffffff",
            command=lambda: self.search_and_recommend(find_least_similar=True),
            cursor="hand2",
            padx=20,
            pady=8,
            relief=tk.FLAT,
            activebackground="#ff4081",
            activeforeground="#ffffff"
        )
        least_similar_btn.pack(side="left", padx=5)
        
        # Options Frame
        options_frame = tk.Frame(row2_frame, bg="#1a1a1a")
        options_frame.pack(side="left", padx=20)
        
        # Number of recommendations
        tk.Label(
            options_frame,
            text="Results:",
            font=("Helvetica", 11),
            bg="#1a1a1a",
            fg="#ffffff"
        ).pack(side="left", padx=(5, 5))
        
        self.num_recommendations = tk.Spinbox(
            options_frame,
            from_=1,
            to=20,
            width=5,
            font=("Helvetica", 10),
            bg="#2a2a2a",
            fg="#ffffff",
            buttonbackground="#1DB954",
            relief=tk.FLAT
        )
        self.num_recommendations.delete(0, "end")
        self.num_recommendations.insert(0, "5")
        self.num_recommendations.pack(side="left", padx=5)

        # Genre Filter Checkbox
        self.same_genre_var = tk.BooleanVar()
        self.genre_check = tk.Checkbutton(
            options_frame,
            text="Same Genre Only",
            variable=self.same_genre_var,
            font=("Helvetica", 11),
            bg="#1a1a1a",
            fg="#ffffff",
            selectcolor="#1a1a1a",
            activebackground="#1a1a1a",
            activeforeground="#1DB954",
            onvalue=True,
            offvalue=False
        )
        self.genre_check.pack(side="left", padx=10)
        
        # Selected Song Frame
        selected_frame = tk.LabelFrame(
            self.root,
            text="Selected Song",
            font=("Helvetica", 13, "bold"),
            bg="#2a2a2a",
            fg="#1DB954",
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=0
        )
        selected_frame.pack(pady=15, padx=30, fill="x")
        
        self.selected_song_label = tk.Label(
            selected_frame,
            text="No song selected",
            font=("Helvetica", 11),
            bg="#2a2a2a",
            fg="#ffffff",
            justify="left",
            anchor="w"
        )
        self.selected_song_label.pack(fill="x")
        
        # Recommendations Frame
        self.rec_frame = tk.LabelFrame(
            self.root,
            text="Recommended Similar Songs",
            font=("Helvetica", 13, "bold"),
            bg="#2a2a2a",
            fg="#1DB954",
            padx=15,
            pady=15,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.rec_frame.pack(pady=15, padx=30, fill="both", expand=True)

        # Create scrolled text widget for recommendations
        self.recommendations_text = scrolledtext.ScrolledText(
            self.rec_frame,
            font=("Helvetica", 10),
            wrap=tk.WORD,
            height=20,
            bg="#1e1e1e",
            fg="#ffffff",
            cursor="arrow",
            relief=tk.FLAT,
            insertbackground="#ffffff"
        )
        self.recommendations_text.pack(fill="both", expand=True)
        
        # Configure text tags for styling
        self.recommendations_text.tag_config("title", font=("Helvetica", 11, "bold"), foreground="#1DB954")
        self.recommendations_text.tag_config("artist", font=("Helvetica", 10), foreground="#cccccc")
        self.recommendations_text.tag_config("similarity", font=("Helvetica", 9), foreground="#888888")
        self.recommendations_text.tag_config("link", font=("Helvetica", 9), foreground="#1DB954", underline=True)
        self.recommendations_text.tag_config("genre", font=("Helvetica", 9, "italic"), foreground="#999999")
        
        # Make links clickable
        self.recommendations_text.tag_bind("link", "<Button-1>", self.open_link)
        self.recommendations_text.tag_bind("link", "<Enter>", lambda e: self.recommendations_text.config(cursor="hand2"))
        self.recommendations_text.tag_bind("link", "<Leave>", lambda e: self.recommendations_text.config(cursor="arrow"))
        
    def on_key_release(self, event):
        """Handle key release events for autocomplete"""
        # Ignore special keys
        if event.keysym in ['Up', 'Down', 'Left', 'Right', 'Return', 'Escape']:
            if event.keysym == 'Escape':
                self.hide_autocomplete()
            return
        
        search_text = self.song_entry.get().strip()
        
        # Hide autocomplete if search text is too short
        if len(search_text) < 2:
            self.hide_autocomplete()
            return
        
        # Find matching songs by track name OR artist name (treat as literal string, not regex)
        track_matches = self.df['track_name'].str.contains(search_text, case=False, na=False, regex=False)
        artist_matches = self.df['artist_name'].str.contains(search_text, case=False, na=False, regex=False)
        matches = self.df[track_matches | artist_matches]
        
        if matches.empty:
            self.hide_autocomplete()
            return
        
        # Limit to top 15 matches for performance
        matches = matches.head(15)
        
        # Update autocomplete listbox
        self.autocomplete_listbox.delete(0, tk.END)
        self.suggestion_data = []
        
        for idx, row in matches.iterrows():
            display_text = f"{row['track_name']} - {row['artist_name']}"
            self.autocomplete_listbox.insert(tk.END, display_text)
            self.suggestion_data.append(row)
        
        # Show autocomplete frame
        self.autocomplete_frame.pack(fill="both", expand=True)
        
    def on_arrow_down(self, event):
        """Handle down arrow key to focus on autocomplete list"""
        if self.autocomplete_frame.winfo_ismapped():
            self.autocomplete_listbox.focus_set()
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(0)
            self.autocomplete_listbox.activate(0)
    
    def on_listbox_click(self, event):
        """Handle mouse click on autocomplete listbox"""
        # Get the index of the item at the click position
        index = self.autocomplete_listbox.nearest(event.y)
        if index >= 0 and index < len(self.suggestion_data):
            self.autocomplete_listbox.selection_clear(0, tk.END)
            self.autocomplete_listbox.selection_set(index)
            self.autocomplete_listbox.activate(index)
            # Process the clicked item
            self.process_listbox_selection(index)
    
    def process_listbox_selection(self, idx=None):
        """Process the selected item from autocomplete"""
        if idx is None:
            selection = self.autocomplete_listbox.curselection()
            if not selection:
                return
            idx = selection[0]
        
        if idx < len(self.suggestion_data):
            selected_song = self.suggestion_data[idx]
            
            # Update entry with selected song name
            self.song_entry.delete(0, tk.END)
            self.song_entry.insert(0, selected_song['track_name'])
            
            # Hide autocomplete
            self.hide_autocomplete()
            
            # Automatically trigger search
            self.song_entry.focus_set()
            self.search_and_recommend()
        
    def on_select_suggestion(self, event):
        """Handle selection from autocomplete list"""
        selection = self.autocomplete_listbox.curselection()
        if not selection:
            return
        
        idx = selection[0]
        selected_song = self.suggestion_data[idx]
        
        # Update entry with selected song name
        self.song_entry.delete(0, tk.END)
        self.song_entry.insert(0, selected_song['track_name'])
        
        # Hide autocomplete
        self.hide_autocomplete()
        
        # Automatically trigger search
        self.song_entry.focus_set()
        self.search_and_recommend()
        
    def hide_autocomplete(self):
        """Hide the autocomplete dropdown"""
        self.autocomplete_frame.pack_forget()
        
    def search_and_recommend(self, find_least_similar=False):
        """Search for the song and find similar recommendations"""
        # Hide autocomplete dropdown
        self.hide_autocomplete()
        
        song_name = self.song_entry.get().strip()
        
        if not song_name:
            messagebox.showwarning("Input Required", "Please enter a song name!")
            return
        
        # Try multiple search strategies for better matching
        # 1. Try exact match first (case-insensitive)
        exact_track = self.df['track_name'].str.lower() == song_name.lower()
        exact_artist = self.df['artist_name'].str.lower() == song_name.lower()
        matches = self.df[exact_track | exact_artist]
        
        # 2. If no exact match, try partial match (escape special regex characters)
        if matches.empty:
            import re
            escaped_name = re.escape(song_name)
            track_matches = self.df['track_name'].str.contains(escaped_name, case=False, na=False, regex=True)
            artist_matches = self.df['artist_name'].str.contains(escaped_name, case=False, na=False, regex=True)
            matches = self.df[track_matches | artist_matches]
        
        if matches.empty:
            messagebox.showerror("Not Found", f"No songs found matching '{song_name}'")
            return
        
        # If multiple matches, use the first one (could be enhanced with a selection dialog)
        if len(matches) > 1:
            # Show info about multiple matches
            match_info = f"Found {len(matches)} matches. Using: {matches.iloc[0]['track_name']}"
            print(match_info)
        
        selected_song = matches.iloc[0]
        selected_idx = matches.index[0]
        
        # Store for comparison
        self.current_input_song = selected_song
        
        # Update selected song display
        self.display_selected_song(selected_song)
        
        # Filter by genre if checkbox is checked
        if self.same_genre_var.get():
            genre = selected_song['genre']
            genre_df = self.df[self.df['genre'] == genre]
            
            if genre_df.empty:
                recommendation_pool = self.df
            else:
                recommendation_pool = genre_df
        else:
            recommendation_pool = self.df
            
        # Extract features for the selected song
        # We need to reshape to (1, -1) for a single sample
        # And we must use the scaler to transform it to match the training data distribution
        song_features_raw = selected_song[self.feature_columns].values.reshape(1, -1)
        song_features = self.scaler.transform(song_features_raw)
        
        # Extract features for the pool
        pool_features_raw = recommendation_pool[self.feature_columns].values
        pool_features = self.scaler.transform(pool_features_raw)
        
        # Get top N recommendations
        try:
            num_recs = int(self.num_recommendations.get())
        except ValueError:
            num_recs = 5

        # Fit NearestNeighbors on the pool features using cosine distance
        nn = NearestNeighbors(metric='cosine')
        nn.fit(pool_features)
        
        # Query for all neighbors (including the query itself)
        distances, indices = nn.kneighbors(song_features, n_neighbors=pool_features.shape[0])
        
        # Convert cosine distance to similarity (cosine similarity = 1 - distance)
        similarities = 1 - distances[0]
        
        if find_least_similar:
            # Sort ascending for least similar (largest distance / smallest similarity)
            # kneighbors returns smallest distance first, so reverse it
            sorted_indices_in_pool = indices[0][::-1]
            sorted_similarities = similarities[::-1]
        else:
            # Sort descending for most similar (smallest distance / largest similarity)
            # kneighbors returns smallest distance first, so keep as is
            sorted_indices_in_pool = indices[0]
            sorted_similarities = similarities
        
        recommendations = []
        count = 0
        
        for i, idx in enumerate(sorted_indices_in_pool):
            candidate_song = recommendation_pool.iloc[idx]
            
            # Skip the song itself
            if candidate_song['track_id'] == selected_song['track_id']:
                continue
                
            recommendations.append({
                'song': candidate_song,
                'score': sorted_similarities[i]
            })
            
            count += 1
            if count >= num_recs:
                break
        
        self.display_recommendations(recommendations, is_least_similar=find_least_similar)
        
    def display_selected_song(self, song):
        """Display the selected song information"""
        song_info = f"{song['track_name']}\n"
        song_info += f"Artist: {song['artist_name']}\n"
        song_info += f"Genre: {song['genre']}\n"
        song_info += f"Popularity: {song['popularity']}"
        
        # Keep selected song text white for visibility on dark background
        self.selected_song_label.config(text=song_info, fg="#ffffff")
        
    def get_similar_songs(self, song_idx, num_recommendations=5):
        """Find similar songs using cosine similarity"""
        # Get the feature vector for the selected song
        song_features = self.normalized_features[song_idx].reshape(1, -1)
        
        # Fit NearestNeighbors on all features
        nn = NearestNeighbors(metric='cosine')
        nn.fit(self.normalized_features)
        
        # Query for neighbors (including the song itself)
        # We ask for num_recommendations + 1 because the first result will be the song itself
        distances, indices = nn.kneighbors(song_features, n_neighbors=num_recommendations + 1)
        
        # Convert cosine distance to similarity
        similarities = 1 - distances[0]
        
        # Get indices of most similar songs (excluding the song itself)
        # kneighbors returns sorted by distance (ascending) -> most similar first
        # indices[0][0] is the song itself, so we skip it
        similar_indices = indices[0][1:]
        similar_scores = similarities[1:]
        
        # Create recommendations list with similarity scores
        recommendations = []
        for i, idx in enumerate(similar_indices):
            song_data = self.df.iloc[idx].copy()
            song_data['similarity_score'] = similar_scores[i]
            recommendations.append(song_data)
        
        return recommendations
        
    def display_recommendations(self, recommendations, is_least_similar=False):
        """Display the recommended songs"""
        # Update frame title
        if is_least_similar:
            self.rec_frame.config(text="Least Similar Songs (Most Different)")
        else:
            self.rec_frame.config(text="Recommended Similar Songs")
            
        self.recommendations_text.delete(1.0, tk.END)
        
        if not recommendations:
            self.recommendations_text.insert(tk.END, "No recommendations found.")
            return
        
        for i, rec_item in enumerate(recommendations, 1):
            song = rec_item['song']
            similarity_score = rec_item['score']
            
            # Song number and title
            self.recommendations_text.insert(tk.END, f"{i}. ", "title")
            self.recommendations_text.insert(tk.END, f"{song['track_name']}\n", "title")
            
            # Artist
            self.recommendations_text.insert(tk.END, f"   Artist: {song['artist_name']}\n", "artist")
            
            # Genre
            self.recommendations_text.insert(tk.END, f"   Genre: {song['genre']}\n", "genre")
            
            # Similarity score
            similarity_pct = similarity_score * 100
            self.recommendations_text.insert(
                tk.END,
                f"   Similarity: {similarity_pct:.1f}%\n",
                "similarity"
            )
            
            # Spotify link
            spotify_link = f"https://open.spotify.com/track/{song['track_id']}"
            self.recommendations_text.insert(tk.END, "   ", "artist")
            
            # Store link for click handler
            link_start = self.recommendations_text.index(tk.INSERT)
            self.recommendations_text.insert(tk.END, "Open in Spotify", "link")
            link_end = self.recommendations_text.index(tk.INSERT)
            
            # Tag the link with the URL
            tag_name = f"link_{i}"
            self.recommendations_text.tag_add(tag_name, link_start, link_end)
            self.recommendations_text.tag_config(tag_name, foreground="#1DB954", underline=True)
            self.recommendations_text.tag_bind(tag_name, "<Button-1>", lambda e, url=spotify_link: webbrowser.open(url))
            self.recommendations_text.tag_bind(tag_name, "<Enter>", lambda e: self.recommendations_text.config(cursor="hand2"))
            self.recommendations_text.tag_bind(tag_name, "<Leave>", lambda e: self.recommendations_text.config(cursor="arrow"))
            
            # Add "See More" button
            self.recommendations_text.insert(tk.END, "   ")
            see_more_btn = tk.Button(
                self.recommendations_text,
                text="See More",
                font=("Helvetica", 8),
                bg="#333333",
                fg="white",
                relief=tk.FLAT,
                command=lambda s=song: self.show_song_details(s, self.current_input_song)
            )
            self.recommendations_text.window_create(tk.END, window=see_more_btn)
            
            self.recommendations_text.insert(tk.END, "\n\n")
        
        # Scroll to top
        self.recommendations_text.see(1.0)
        
    def open_link(self, event):
        """Handle link clicks"""
        # Get the tag at the click position
        tags = self.recommendations_text.tag_names(tk.CURRENT)
        for tag in tags:
            if tag.startswith("link_"):
                # The URL is bound to the tag via the lambda in display_recommendations
                pass


    def show_song_details(self, song, comparison_song=None):
        """Show detailed view with radar chart"""
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Details: {song['track_name']}")
        details_window.geometry("800x600")
        details_window.configure(bg="#1a1a1a")
        
        # Header
        header_frame = tk.Frame(details_window, bg="#1a1a1a")
        header_frame.pack(pady=20, padx=20, fill="x")
        
        tk.Label(
            header_frame,
            text=song['track_name'],
            font=("Helvetica", 20, "bold"),
            bg="#1a1a1a",
            fg="#1DB954"
        ).pack()
        
        tk.Label(
            header_frame,
            text=f"by {song['artist_name']}",
            font=("Helvetica", 14),
            bg="#1a1a1a",
            fg="#ffffff"
        ).pack()
        
        # Radar Chart
        features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
        
        # Prepare data for chart
        values = [song[f] for f in features]
        values += values[:1]  # Close the polygon
        
        angles = [n / float(len(features)) * 2 * np.pi for n in range(len(features))]
        angles += angles[:1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # Plot selected song
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=song['track_name'], color='#1DB954')
        ax.fill(angles, values, '#1DB954', alpha=0.25)
        
        # Plot comparison song (input song) if provided
        if comparison_song is not None:
            comp_values = [comparison_song[f] for f in features]
            comp_values += comp_values[:1]
            ax.plot(angles, comp_values, linewidth=2, linestyle='solid', label="Input Song", color='#ffffff')
            ax.fill(angles, comp_values, '#ffffff', alpha=0.1)
        
        # Fix axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, color='white', size=10)
        
        # Fix radial labels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="#888888", size=8)
        plt.ylim(0, 1)
        
        # Remove spines
        ax.spines['polar'].set_visible(False)
        
        # Grid color
        ax.grid(color='#333333')
        
        # Legend
        if comparison_song is not None:
            legend = ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), facecolor='#1a1a1a', edgecolor='#333333')
            for text in legend.get_texts():
                text.set_color("white")
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=details_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)
        
        # Close button
        tk.Button(
            details_window,
            text="Close",
            command=details_window.destroy,
            bg="#333333",
            fg="white",
            font=("Helvetica", 10),
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(pady=20)

def main():
    root = tk.Tk()
    app = SongRecommenderApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
