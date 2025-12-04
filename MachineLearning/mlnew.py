import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('SpotifyFeatures.csv')  # Replace with your file path

# Data Cleaning & Preprocessing
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values if any
df = df.dropna()

df_fixed = df.drop(['track_id', 'track_name', 'artist_name'], axis=1)

# Define feature columns BEFORE encoding
feature_columns = [
    'popularity', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'instrumentalness', 'key', 'liveness', 'loudness',
    'mode', 'speechiness', 'tempo', 'time_signature', 'valence'
]

# Encode categorical features
categorical_columns = ['key', 'genre', 'mode', 'time_signature']

for col in categorical_columns:
    df_fixed[col] = pd.factorize(df_fixed[col])[0]

# Create genre_encoded column for target
df_fixed['genre_encoded'] = df_fixed['genre']

# Prepare features and target
X = df_fixed[feature_columns]
y = df_fixed['genre_encoded']

# Create label encoder for genre names
label_encoder = LabelEncoder()
label_encoder.fit(df['genre'])  # Fit on original genre names

print("\nGenres in dataset:")
for i, genre in enumerate(label_encoder.classes_):
    print(f"{i}: {genre}")

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")
print(f"Train/Test ratio: {X_train.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%} / {X_test.shape[0] / (X_train.shape[0] + X_test.shape[0]):.1%}")

# Initialize and train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)

# Calculate training and test scores
train_score = rf_model.score(X_train, y_train)
test_score = rf_model.score(X_test, y_test)

# Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== MODEL PERFORMANCE ===")
print(f"Training Score: {train_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Number of classes: {len(label_encoder.classes_)}")

print("\n=== SIMPLIFIED GENRE PERFORMANCE REPORT ===")
print("=" * 60)

# Get the detailed classification report and make it readable
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

# Convert to DataFrame for better formatting
report_df = pd.DataFrame(report_dict).transpose()

# Add a column for what each metric means
report_df['Metric Meaning'] = ''
report_df.loc['accuracy', 'Metric Meaning'] = 'Overall correct predictions'
report_df.loc['macro avg', 'Metric Meaning'] = 'Average across all genres (treating each equally)'
report_df.loc['weighted avg', 'Metric Meaning'] = 'Average across all genres (weighted by genre size)'

# For individual genres, explain precision and recall
for genre in label_encoder.classes_:
    if genre in report_df.index:
        report_df.loc[genre, 'Metric Meaning'] = f'Precision: How often {genre} predictions were correct | Recall: How many actual {genre} songs were found'

# Display the simplified report
print("\nOVERALL PERFORMANCE:")
print(f"• Overall Accuracy: {report_df.loc['accuracy', 'f1-score']:.1%} of songs were correctly classified")
print(f"• Average Performance (all genres equal): {report_df.loc['macro avg', 'f1-score']:.1%}")
print(f"• Average Performance (weighted by genre size): {report_df.loc['weighted avg', 'f1-score']:.1%}")

print(f"\nTOP 5 BEST PERFORMING GENRES:")
# Get top 5 genres by F1-score (excluding averages)
genre_scores = report_df.loc[label_encoder.classes_].sort_values('f1-score', ascending=False).head()
for i, (genre, row) in enumerate(genre_scores.iterrows(), 1):
    print(f"{i}. {genre}:")
    print(f"   • Precision: {row['precision']:.1%} - When we predict {genre}, we're correct {row['precision']:.1%} of the time")
    print(f"   • Recall: {row['recall']:.1%} - We find {row['recall']:.1%} of all actual {genre} songs")
    print(f"   • F1-Score: {row['f1-score']:.1%} - Overall balance of precision and recall")

print(f"\nBOTTOM 5 WORST PERFORMING GENRES:")
# Get bottom 5 genres by F1-score (excluding averages)
genre_scores_bottom = report_df.loc[label_encoder.classes_].sort_values('f1-score', ascending=True).head()
for i, (genre, row) in enumerate(genre_scores_bottom.iterrows(), 1):
    print(f"{i}. {genre}:")
    print(f"   • Precision: {row['precision']:.1%} - When we predict {genre}, we're correct {row['precision']:.1%} of the time")
    print(f"   • Recall: {row['recall']:.1%} - We find {row['recall']:.1%} of all actual {genre} songs")
    print(f"   • F1-Score: {row['f1-score']:.1%} - Overall balance of precision and recall")

print(f"\nKEY METRICS EXPLAINED:")
print("• PRECISION: When the model says a song is a certain genre, how often is it right?")
print("• RECALL: Of all the songs that are actually a certain genre, how many did the model find?") 
print("• F1-SCORE: Balanced measure of both precision and recall (higher is better)")
print("• SUPPORT: How many songs of this genre were in the test set")

# Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print("Most important audio features for genre prediction:")
for i, row in feature_importance.head(5).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.1%}")

# Visualization
plt.figure(figsize=(15, 10))

# Feature Importance Plot
plt.subplot(2, 2, 1)
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Top Audio Features for Genre Prediction')
plt.xlabel('Importance (%)')

# Confusion Matrix
plt.subplot(2, 2, 2)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=False, cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix - Which genres get confused?')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Genre Performance Chart
plt.subplot(2, 2, 3)
top_genres = report_df.loc[label_encoder.classes_].nlargest(10, 'f1-score')
plt.barh(range(len(top_genres)), top_genres['f1-score'])
plt.yticks(range(len(top_genres)), top_genres.index)
plt.title('Top 10 Genres by Prediction Accuracy')
plt.xlabel('F1-Score (Higher is Better)')

plt.tight_layout()
plt.show()

# Predict on new data function
def predict_genre(audio_features_dict, model=rf_model, encoder=label_encoder):
    """
    Predict genre for new audio features
    audio_features_dict should contain all features
    """
    # Create feature array in correct order
    features = np.array([[audio_features_dict[col] for col in feature_columns]])

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    # Get results
    predicted_genre = encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]

    print(f"PREDICTION RESULT")
    print(f"Predicted genre: {predicted_genre}")
    print(f"Confidence: {confidence:.1%}")
    
    if confidence > 0.7:
        print("High confidence prediction")
    elif confidence > 0.5:
        print("Moderate confidence prediction")
    else:
        print("Low confidence prediction")

    # Show top 3 probabilities
    print("\nTop 3 possible genres:")
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    for i, idx in enumerate(top_3_indices, 1):
        print(f"{i}. {encoder.classes_[idx]}: {probabilities[idx]:.1%}")

    return predicted_genre, confidence

# Example usage:
# sample_features = {
#     'popularity': 75, 'acousticness': 0.2, 'danceability': 0.8,
#     'duration_ms': 200000, 'energy': 0.9, 'instrumentalness': 0.1,
#     'key': 5, 'liveness': 0.1, 'loudness': -5.0, 'mode': 1,
#     'speechiness': 0.05, 'tempo': 120.0, 'time_signature': 4, 'valence': 0.8
# }
# predict_genre(sample_features)
