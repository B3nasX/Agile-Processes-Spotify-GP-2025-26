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