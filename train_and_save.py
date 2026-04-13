import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 55)
print("   TRAINING MOVIE RATING PREDICTOR")
print("=" * 55)

# ── REAL-WORLD STYLE MOVIE DATASET ────────────────
# Simulates what a streaming platform collects:
# user behavior + movie metadata + rating given
np.random.seed(42)

movies = [
    # title,              genre,      language, release_year, duration, imdb_base
    ("Inception",         "Sci-Fi",   "English", 2010, 148, 8.8),
    ("The Dark Knight",   "Action",   "English", 2008, 152, 9.0),
    ("Parasite",          "Thriller", "Korean",  2019, 132, 8.6),
    ("3 Idiots",          "Comedy",   "Hindi",   2009, 170, 8.4),
    ("Interstellar",      "Sci-Fi",   "English", 2014, 169, 8.6),
    ("Avengers Endgame",  "Action",   "English", 2019, 181, 8.4),
    ("Dangal",            "Drama",    "Hindi",   2016, 161, 8.3),
    ("The Godfather",     "Drama",    "English", 1972, 175, 9.2),
    ("Spirited Away",     "Fantasy",  "Japanese",2001, 125, 8.6),
    ("Pulp Fiction",      "Thriller", "English", 1994, 154, 8.9),
    ("Sholay",            "Action",   "Hindi",   1975, 204, 8.1),
    ("Your Name",         "Fantasy",  "Japanese",2016, 106, 8.4),
    ("Get Out",           "Thriller", "English", 2017, 104, 7.7),
    ("PK",                "Comedy",   "Hindi",   2014, 153, 8.1),
    ("Whiplash",          "Drama",    "English", 2014, 107, 8.5),
    ("Baahubali 2",       "Action",   "Telugu",  2017, 167, 8.2),
    ("The Matrix",        "Sci-Fi",   "English", 1999, 136, 8.7),
    ("Zindagi Na Milegi", "Drama",    "Hindi",   2011, 155, 8.2),
    ("Everything Everywhere","Sci-Fi","English", 2022, 139, 7.8),
    ("RRR",               "Action",   "Telugu",  2022, 187, 7.8),
]

rows = []
n_users = 300

for user_id in range(1, n_users + 1):
    # Each user has preferences
    fav_genre    = np.random.choice(["Action","Drama","Sci-Fi","Thriller","Comedy","Fantasy"])
    fav_language = np.random.choice(["English","Hindi","Korean","Japanese","Telugu"])
    age          = np.random.randint(18, 60)
    watches_old  = np.random.choice([True, False])  # likes old movies?

    # Each user rates 5-12 random movies
    n_ratings  = np.random.randint(5, 13)
    movie_idxs = np.random.choice(len(movies), n_ratings, replace=False)

    for idx in movie_idxs:
        title, genre, lang, year, duration, imdb = movies[idx]

        # Rating logic — preferences drive rating
        base = imdb * 0.6

        if genre == fav_genre:        base += 1.2
        if lang  == fav_language:     base += 0.8
        if watches_old and year<2000: base += 0.5
        if age < 25 and year > 2015:  base += 0.4
        if duration > 160:            base -= 0.2

        # Add noise
        rating = np.clip(base + np.random.normal(0, 0.4), 1.0, 10.0)

        rows.append({
            "user_id"        : user_id,
            "title"          : title,
            "genre"          : genre,
            "language"       : lang,
            "release_year"   : year,
            "duration_mins"  : duration,
            "imdb_score"     : imdb,
            "user_age"       : age,
            "fav_genre"      : fav_genre,
            "fav_language"   : fav_language,
            "genre_match"    : int(genre == fav_genre),
            "lang_match"     : int(lang  == fav_language),
            "user_rating"    : round(rating, 1)
        })

df = pd.DataFrame(rows)
print(f"Dataset: {len(df):,} ratings from {n_users} users on {len(movies)} movies")
print(f"\nSample data:")
print(df[["user_id","title","genre","user_age","user_rating"]].head(8).to_string(index=False))
print(f"\nRating distribution:")
print(f"   Min   : {df['user_rating'].min():.1f}")
print(f"   Avg   : {df['user_rating'].mean():.2f}")
print(f"   Max   : {df['user_rating'].max():.1f}")

# ── ENCODE ────────────────────────────────────────
le_genre = LabelEncoder()
le_lang  = LabelEncoder()
le_fav_g = LabelEncoder()
le_fav_l = LabelEncoder()

df["genre_enc"]    = le_genre.fit_transform(df["genre"])
df["lang_enc"]     = le_lang.fit_transform(df["language"])
df["fav_genre_enc"]= le_fav_g.fit_transform(df["fav_genre"])
df["fav_lang_enc"] = le_fav_l.fit_transform(df["fav_language"])

# ── TRAIN ─────────────────────────────────────────
features = [
    "genre_enc", "lang_enc", "release_year",
    "duration_mins", "imdb_score", "user_age",
    "fav_genre_enc", "fav_lang_enc",
    "genre_match", "lang_match"
]

X = df[features]
y = df["user_rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining GradientBoostingRegressor...")
model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1,
    max_depth=4, random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
print(f"   R² Score  : {r2:.4f}")
print(f"   Avg Error : {mae:.3f} stars")

# ── SAVE ──────────────────────────────────────────
joblib.dump(model,    "model.pkl")
joblib.dump(le_genre, "le_genre.pkl")
joblib.dump(le_lang,  "le_lang.pkl")
joblib.dump(le_fav_g, "le_fav_genre.pkl")
joblib.dump(le_fav_l, "le_fav_lang.pkl")

print("\nmodel.pkl + encoders saved")
print("   API is ready to serve predictions!")
