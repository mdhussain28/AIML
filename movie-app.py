import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# ── LOAD MODEL + ENCODERS ─────────────────────────
print("Loading model and encoders...")
model    = joblib.load("model.pkl")
le_genre = joblib.load("le_genre.pkl")
le_lang  = joblib.load("le_lang.pkl")
le_fav_g = joblib.load("le_fav_genre.pkl")
le_fav_l = joblib.load("le_fav_lang.pkl")

VALID_GENRES    = list(le_genre.classes_)
VALID_LANGUAGES = list(le_lang.classes_)
print(f"Ready — Genres: {VALID_GENRES}")
print(f"Ready — Languages: {VALID_LANGUAGES}")

# ── APP ───────────────────────────────────────────
app = FastAPI(
    title="🎬 Movie Rating Prediction API",
    description="Predict how much a user will enjoy a movie — like Netflix does",
    version="1.0.0"
)

# ── SCHEMAS ───────────────────────────────────────
class RatingRequest(BaseModel):
    # Movie details
    genre        : str    # "Action", "Drama", "Sci-Fi" etc
    language     : str    # "English", "Hindi", "Korean" etc
    release_year : int    # 1970 - 2024
    duration_mins: int    # movie length in minutes
    imdb_score   : float  # IMDb rating 1-10

    # User profile
    user_age     : int    # user's age
    fav_genre    : str    # user's favourite genre
    fav_language : str    # user's favourite language

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "genre"        : "Sci-Fi",
                "language"     : "English",
                "release_year" : 2014,
                "duration_mins": 169,
                "imdb_score"   : 8.6,
                "user_age"     : 28,
                "fav_genre"    : "Sci-Fi",
                "fav_language" : "English"
            }]
        }
    }

# ── HELPER ────────────────────────────────────────
def rating_to_verdict(rating: float) -> str:
    if rating >= 8.5: return "You'll absolutely love it!"
    if rating >= 7.5: return "You'll really enjoy this"
    if rating >= 6.5: return "Worth watching"
    if rating >= 5.0: return "It's okay, your call"
    return "Probably skip this one"

# ── ROUTES ────────────────────────────────────────
@app.get("/")
def root():
    return {
        "api"      : "Movie Rating Prediction API",
        "status"   : "running",
        "docs"     : "/docs",
        "endpoints": ["/predict", "/recommend", "/health", "/movies/genres"]
    }

@app.get("/health")
def health():
    return {
        "status"    : "healthy",
        "model"     : "GradientBoostingRegressor",
        "timestamp" : datetime.now().isoformat()
    }

@app.get("/movies/genres")
def get_genres():
    return {
        "genres"    : VALID_GENRES,
        "languages" : VALID_LANGUAGES
    }

@app.post("/predict")
def predict_rating(req: RatingRequest):
    # Validate genre and language
    if req.genre not in VALID_GENRES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid genre '{req.genre}'. Valid: {VALID_GENRES}"
        )
    if req.language not in VALID_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language '{req.language}'. Valid: {VALID_LANGUAGES}"
        )

    try:
        # Encode inputs
        genre_enc    = le_genre.transform([req.genre])[0]
        lang_enc     = le_lang.transform([req.language])[0]
        fav_g_enc    = le_fav_g.transform([req.fav_genre])[0]
        fav_l_enc    = le_fav_l.transform([req.fav_language])[0]
        genre_match  = int(req.genre == req.fav_genre)
        lang_match   = int(req.language == req.fav_language)

        features = np.array([[
            genre_enc, lang_enc, req.release_year,
            req.duration_mins, req.imdb_score, req.user_age,
            fav_g_enc, fav_l_enc, genre_match, lang_match
        ]])

        predicted_rating = float(model.predict(features)[0])
        predicted_rating = round(np.clip(predicted_rating, 1.0, 10.0), 2)

        return {
            "predicted_rating" : predicted_rating,
            "out_of"           : 10,
            "stars"            : "⭐" * round(predicted_rating / 2),
            "verdict"          : rating_to_verdict(predicted_rating),
            "genre_match"      : bool(genre_match),
            "language_match"   : bool(lang_match),
            "timestamp"        : datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
def recommend(user_age: int, fav_genre: str, fav_language: str):
    """Score all movies in catalogue for this user and rank them"""

    catalogue = [
        ("Inception",        "Sci-Fi",   "English", 2010, 148, 8.8),
        ("The Dark Knight",  "Action",   "English", 2008, 152, 9.0),
        ("Parasite",         "Thriller", "Korean",  2019, 132, 8.6),
        ("3 Idiots",         "Comedy",   "Hindi",   2009, 170, 8.4),
        ("Interstellar",     "Sci-Fi",   "English", 2014, 169, 8.6),
        ("Dangal",           "Drama",    "Hindi",   2016, 161, 8.3),
        ("The Godfather",    "Drama",    "English", 1972, 175, 9.2),
        ("Spirited Away",    "Fantasy",  "Japanese",2001, 125, 8.6),
        ("Pulp Fiction",     "Thriller", "English", 1994, 154, 8.9),
        ("RRR",              "Action",   "Telugu",  2022, 187, 7.8),
    ]

    results = []
    for title, genre, lang, year, dur, imdb in catalogue:
        try:
            g_enc  = le_genre.transform([genre])[0]
            l_enc  = le_lang.transform([lang])[0]
            fg_enc = le_fav_g.transform([fav_genre])[0]
            fl_enc = le_fav_l.transform([fav_language])[0]
            gm     = int(genre == fav_genre)
            lm     = int(lang  == fav_language)

            feat   = np.array([[g_enc, l_enc, year, dur, imdb,
                                 user_age, fg_enc, fl_enc, gm, lm]])
            rating = float(model.predict(feat)[0])
            rating = round(np.clip(rating, 1.0, 10.0), 2)

            results.append({
                "title"  : title,
                "genre"  : genre,
                "language": lang,
                "predicted_rating": rating,
                "verdict": rating_to_verdict(rating)
            })
        except:
            continue

    results.sort(key=lambda x: x["predicted_rating"], reverse=True)

    return {
        "user_profile" : {
            "age"         : user_age,
            "fav_genre"   : fav_genre,
            "fav_language": fav_language
        },
        "top_picks"    : results[:5],
        "avoid"        : results[-2:]
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
