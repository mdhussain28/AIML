import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)

movies = [
    # title                  genre       language   year  mins  imdb
    ("The Dark Knight",    "Action",   "English", 2008, 152, 9.0),
    ("Inception",          "Sci-Fi",   "English", 2010, 148, 8.8),
    ("Interstellar",       "Sci-Fi",   "English", 2014, 169, 8.6),
    ("Pulp Fiction",       "Thriller", "English", 1994, 154, 8.9),
    ("The Godfather",      "Drama",    "English", 1972, 175, 9.2),
    ("Whiplash",           "Drama",    "English", 2014, 107, 8.5),
    ("The Matrix",         "Sci-Fi",   "English", 1999, 136, 8.7),
    ("Avengers Endgame",   "Action",   "English", 2019, 181, 8.4),
    # Hindi movies
    ("Sholay",             "Action",   "Hindi",   1975, 204, 8.1),
    ("Dangal",             "Drama",    "Hindi",   2016, 161, 8.3),
    ("3 Idiots",           "Comedy",   "Hindi",   2009, 170, 8.4),
    ("PK",                 "Comedy",   "Hindi",   2014, 153, 8.1),
    ("Zindagi Na Milegi",  "Drama",    "Hindi",   2011, 155, 8.2),
    ("Taare Zameen Par",   "Drama",    "Hindi",   2007, 162, 8.5),
    ("War",                "Action",   "Hindi",   2019, 154, 5.1),
    ("Pathaan",            "Action",   "Hindi",   2023, 146, 5.9),
    ("Bhaag Milkha Bhaag", "Drama",    "Hindi",   2013, 186, 8.2),
    ("Dil Chahta Hai",     "Comedy",   "Hindi",   2001, 183, 8.1),
    # Korean
    ("Parasite",           "Thriller", "Korean",  2019, 132, 8.6),
    ("Oldboy",             "Thriller", "Korean",  2003, 120, 8.1),
    # Japanese
    ("Spirited Away",      "Fantasy",  "Japanese",2001, 125, 8.6),
    ("Your Name",          "Fantasy",  "Japanese",2016, 106, 8.4),
    # Telugu
    ("Baahubali 2",        "Action",   "Telugu",  2017, 167, 8.2),
    ("RRR",                "Action",   "Telugu",  2022, 187, 7.8),
]

rows = []
for user_id in range(1, 2000):
    fav_genre    = np.random.choice(["Action","Drama","Sci-Fi","Thriller","Comedy","Fantasy"])
    fav_language = np.random.choice(["English","Hindi","Korean","Japanese","Telugu"])
    age          = np.random.randint(18, 60)

    movie_idxs = np.random.choice(len(movies), np.random.randint(10, 20), replace=False)

    for idx in movie_idxs:
        title, genre, lang, year, duration, imdb = movies[idx]

        # Start from 5.0 baseline — not IMDb
        # IMDb is just a small bonus, not the base
        base = 5.0

        # ── LANGUAGE match is the strongest signal ──
        if lang == fav_language:
            base += 3.5       # big boost for native language
        else:
            base -= 2.0       # penalty for foreign language

        # ── GENRE match is second strongest ──
        if genre == fav_genre:
            base += 2.5       # boost for favourite genre
        else:
            base -= 1.0       # penalty for non-favourite genre

        # IMDb is a small quality bonus on top
        base += (imdb - 8.0) * 0.3

        # Age preference
        if age < 30 and year > 2010:
            base += 0.3
        if age > 45 and year < 2000:
            base += 0.3

        # Long movie slight penalty
        if duration > 180:
            base -= 0.2

        rating = float(np.clip(base + np.random.normal(0, 0.15), 1.0, 10.0))

        rows.append({
            "genre"        : genre,
            "language"     : lang,
            "release_year" : year,
            "duration_mins": duration,
            "imdb_score"   : imdb,
            "user_age"     : age,
            "fav_genre"    : fav_genre,
            "fav_language" : fav_language,
            "genre_match"  : int(genre == fav_genre),
            "lang_match"   : int(lang == fav_language),
            "user_rating"  : round(rating, 1)
        })

df = pd.DataFrame(rows)

print("=" * 55)
print("   TRAINING MOVIE RATING PREDICTOR")
print("=" * 55)
print(f"Total ratings : {len(df):,}")

# ── SANITY CHECK ──────────────────────────────────
# Language match must be the biggest driver
lang_yes = df[df['lang_match']==1]['user_rating'].mean()
lang_no  = df[df['lang_match']==0]['user_rating'].mean()
gen_yes  = df[df['genre_match']==1]['user_rating'].mean()
gen_no   = df[df['genre_match']==0]['user_rating'].mean()

print(f"\nLanguage match avg   : {lang_yes:.2f}")
print(f"Language no-match avg: {lang_no:.2f}")
print(f"Language difference  : {lang_yes - lang_no:.2f}  <- must be 3.0+")
print(f"\nGenre match avg      : {gen_yes:.2f}")
print(f"Genre no-match avg   : {gen_no:.2f}")
print(f"Genre difference     : {gen_yes - gen_no:.2f}  <- must be 2.0+")

# ── ENCODE ────────────────────────────────────────
le_genre = LabelEncoder()
le_lang  = LabelEncoder()
le_fav_g = LabelEncoder()
le_fav_l = LabelEncoder()

df["genre_enc"]     = le_genre.fit_transform(df["genre"])
df["lang_enc"]      = le_lang.fit_transform(df["language"])
df["fav_genre_enc"] = le_fav_g.fit_transform(df["fav_genre"])
df["fav_lang_enc"]  = le_fav_l.fit_transform(df["fav_language"])

features = [
    "genre_enc","lang_enc","release_year","duration_mins",
    "imdb_score","user_age","fav_genre_enc","fav_lang_enc",
    "genre_match","lang_match"
]

X = df[features]
y = df["user_rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining model...")
model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05,
    max_depth=5, random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"R²  : {r2_score(y_test, preds):.4f}")
print(f"MAE : {mean_absolute_error(y_test, preds):.3f} stars")

# ── QUICK PREDICTION TEST ─────────────────────────
# Manually verify: Hindi fan should rate Hindi movie higher
print("\n── Sanity prediction test ──")
test_cases = [
    ("Hindi Action fan   → Sholay (Hindi/Action)",   "Action","Hindi",  1975,204,8.1,"Action","Hindi"),
    ("Hindi Action fan   → Dark Knight (Eng/Action)","Action","English",2008,152,9.0,"Action","Hindi"),
    ("English Drama fan  → Godfather (Eng/Drama)",   "Drama", "English",1972,175,9.2,"Drama", "English"),
    ("English Drama fan  → Dangal (Hindi/Drama)",    "Drama", "Hindi",  2016,161,8.3,"Drama", "English"),
]

for label, genre, lang, year, dur, imdb, fg, fl in test_cases:
    ge  = le_genre.transform([genre])[0]
    le  = le_lang.transform([lang])[0]
    fge = le_fav_g.transform([fg])[0]
    fle = le_fav_l.transform([fl])[0]
    gm  = int(genre == fg)
    lm  = int(lang  == fl)
    f   = np.array([[ge, le, year, dur, imdb, 28, fge, fle, gm, lm]])
    r   = float(np.clip(model.predict(f)[0], 1, 10))
    print(f"   {label:45} → {r:.2f}")

joblib.dump(model,    "model.pkl")
joblib.dump(le_genre, "le_genre.pkl")
joblib.dump(le_lang,  "le_lang.pkl")
joblib.dump(le_fav_g, "le_fav_genre.pkl")
joblib.dump(le_fav_l, "le_fav_lang.pkl")
print("\nAll files saved. API ready!")
