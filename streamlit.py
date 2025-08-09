# streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle

# Custom imports
from utils.functions import RecommenderUtils

# Load mappings
with open('.models/user_mapping.pkl', 'rb') as f:
    user_userEncoded = pickle.load(f)
with open('.models/anime_mapping.pkl', 'rb') as f:
    anime_animeEncoded = pickle.load(f)
with open('.models/rev_user_mapping.pkl', 'rb') as f:
    userEncoded_user = pickle.load(f)
with open('.models/rev_anime_mapping.pkl', 'rb') as f:
    animeEncoded_anime = pickle.load(f)

# Load trained model
model = keras.models.load_model('.models/recommendation_anime_model.keras')

# Load datasets
ratings_df = pd.read_csv('.data/ratings_df.csv').drop(['Unnamed: 0'], axis=1)
data = pd.read_csv('.data/cleaned_animelist.csv').drop(['Unnamed: 0'], axis=1)
test_df = pd.read_csv('.data/test_df.csv').drop(['Unnamed: 0'], axis=1)

# Extract normalized embeddings
def extract_weights(name, model):
    weights = model.get_layer(name).get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    return weights

anime_weights = extract_weights('anime_embedding', model)
user_weights = extract_weights('user_embedding', model)

# Create recommender instance
rc = RecommenderUtils(
    model=model,
    anime_animeEncoded=anime_animeEncoded,
    anime_weights=anime_weights,
    animeEncoded_anime=animeEncoded_anime,
    user_userEncoded=user_userEncoded,
    userEncoded_user=userEncoded_user,
    data=data,
    ratings_df=ratings_df,
    user_weights=user_weights
)

st.set_page_config(page_title="Anime Recommendation System", layout="centered")
st.title("Similar Anime Recommendation System")

anime_input = st.text_input("Enter an anime title:")

if anime_input:
    try:
        recs_df = rc.find_similar_animes(anime_input, n=10)  # This should return a DataFrame
        st.subheader(f"Recommendations similar to: {anime_input}")
        st.dataframe(recs_df)
    except Exception as e:
        st.error(f"Error: {str(e)}. Please make sure the anime title exists in the database.")
