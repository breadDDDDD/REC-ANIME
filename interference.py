# Lib
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Add, Activation, Lambda, BatchNormalization, Concatenate, Dropout, Input, Embedding, Dot, Reshape, Dense, Flatten
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
import pickle

#get the functions
from utils.functions import RecommenderUtils
from utils.cosine import UserEmbeddingMatcher

'''
This is a inference example script of this model,

ratings_df = users used for training the model
test_df = an example of users outside of training
data = full of anime data like genre, title, and etc

There are 2 main uses, find an anime similar to the anime
you inputted or get a recommendation based on a userID

The model is trained based on UserID and AnimeID, for more
information of the steps, see the jupter notebook
'''

# embedder
with open('.models/user_mapping.pkl', 'rb') as f:
    user_userEncoded = pickle.load(f)
with open('.models/anime_mapping.pkl', 'rb') as f:
    anime_animeEncoded = pickle.load(f)
with open('.models/rev_user_mapping.pkl', 'rb') as f:
    userEncoded_user = pickle.load(f)
with open('.models/rev_anime_mapping.pkl', 'rb') as f:
    animeEncoded_anime = pickle.load(f)

n_users = len(user_userEncoded)
n_animes = len(anime_animeEncoded)

#model
model = keras.models.load_model('.models/recommendation_anime_model.keras')
# model.summary()

#load embedded data
ratings_df = pd.read_csv('.data/ratings_df.csv')
data = pd.read_csv('.data/cleaned_animelist.csv')
test_df = pd.read_csv('.data/test_df.csv')
ratings_df = ratings_df.drop(['Unnamed: 0'],axis=1)
data = data.drop(['Unnamed: 0'],axis=1)
test_df = test_df.drop(['Unnamed: 0'],axis=1)


#extract the weights for user and anime
def extract_weights(name, model):
    weights = model.get_layer(name).get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    return weights

anime_weights = extract_weights('anime_embedding', model)
user_weights = extract_weights('user_embedding', model)

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

mt = UserEmbeddingMatcher(
    anime_animeEncoded, 
    anime_weights, 
    user_weights, 
    userEncoded_user, 
    test_df)

print('Objective of User')
print(''' List :
    1. Find similar animes
    2. Get recommendations based on userID
      ''')
print()
objective = input('Enter your objective : ')

# get similar animes
if objective == '1':
    anime = input('Enter anime name : ')
    recomms_anime = rc.find_similar_animes(anime, n=10)
    print(recomms_anime)

# get recommendations based on userID   
elif objective == '2':
    test_user = input('Enter user ID : ')
    test_user = int(test_user)
    
    # check if the user is in ratings_df
    if test_user in ratings_df['userID'].values:
        test_user = test_user
    else:
        close, score_close = mt.get_similar_test(test_user)
        test_user = close
    
    #the user prefs
    similar_users = rc.find_similar_users(int(test_user), 
                                    n=5, 
                                    neg=False)
    similar_users = similar_users[similar_users.similarity > 0.4]
    similar_users = similar_users[similar_users.similar_users != test_user]
    user_pref = rc.get_user_preferences(test_user, verbose=0)

    #get recoms
    def get_recommended_animes(rc, similar_users, n=10):
        recommended_animes = []
        anime_list = []

        for user_id in similar_users.similar_users.values:
            pref_list = rc.get_user_preferences( int(user_id), verbose=0)
            pref_list = pref_list[~pref_list.title.isin(user_pref.title.values)]
            anime_list.append(pref_list.title.values)

        anime_list = pd.DataFrame(anime_list)
        sorted_list = pd.DataFrame(pd.Series(anime_list.values.ravel()).value_counts()).head(n)

        for anime_name in sorted_list.index:
            if isinstance(anime_name, str):
                frame = rc.getAnimeFrame( anime_name)
                if not frame.empty:
                    try:
                        genre = frame['genres'].values[0]
                        score = frame['score'].values[0]
                        episodes = frame['episodes'].values[0]

                        recommended_animes.append({
                            "anime_name": anime_name,
                            "genre": genre,
                            "score": score,
                            "episodes": episodes
                        })
                    except Exception as e:
                        print(f"Error processing '{anime_name}': {e}")

        return pd.DataFrame(recommended_animes)

    recommended_animes = get_recommended_animes(rc,similar_users, n=5)

    print(recommended_animes)
    
else:
    print('wrong objective')