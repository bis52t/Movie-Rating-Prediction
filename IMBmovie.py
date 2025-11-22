import pandas as pd

# Load the dataset
file_path = r"C:/Users/tusha/OneDrive/Documents/Titanic_project/IMDb Movies India.csv"
df = pd.read_csv(file_path, encoding="latin1")

# Clean Year
df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Clean Duration
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Clean Votes
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')

# Split Genre
df['Genre'] = df['Genre'].astype(str).str.split(', ')

# Clean text columns
text_cols = ['Name', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
for col in text_cols:df[col] = df[col].astype(str).str.strip()
df[col] = df[col].astype(str).str.strip()


# Print results
print("----- HEAD -----")
print(df.head())

print("\n----- INFO -----")
print(df.info())
print("\n----- TOP 10 HIGHEST RATED MOVIES -----")
top_rated = df.sort_values(by='Rating', ascending=False).head(10)
print(top_rated[['Name', 'Year', 'Rating', 'Votes']])

print("\n----- TOP 10 MOST VOTED MOVIES -----")
top_voted = df.sort_values(by='Votes', ascending=False).head(10)
print(top_voted[['Name', 'Year', 'Rating', 'Votes']])
print("\n----- BUILDING MOVIE RATING PREDICTION MODEL -----")

print("\n----- BUILDING MOVIE RATING PREDICTION MODEL -----")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Keep only rows with Rating
df_model = df.dropna(subset=["Rating"]).copy().reset_index(drop=True)

# --- GENRE ENCODING ---
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df_model['Genre'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

# --- TARGET ENCODE Director & Actors ---
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df_model[col + "_enc"] = df_model.groupby(col)['Rating'].transform("mean")

# Combine all features
features = pd.concat([
    df_model[['Year', 'Duration', 'Votes',
              'Director_enc', 'Actor 1_enc', 'Actor 2_enc', 'Actor 3_enc']],
    genre_df
], axis=1)

target = df_model['Rating']

print("Features shape:", features.shape)
print("Target shape:", target.shape)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, y_pred) ** 0.5   # FIXED
r2 = r2_score(y_test, y_pred)

print("\n----- MODEL PERFORMANCE -----")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")


# Example prediction
test_example = X_test.iloc[0:1]
prediction = model.predict(test_example)

print("\n----- SAMPLE PREDICTION -----")
print("Movie:", df_model.iloc[test_example.index[0]]["Name"])
print("Actual Rating:", y_test.iloc[0])
print("Predicted Rating:", round(prediction[0], 2))
