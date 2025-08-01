import numpy as np
import pandas as pd
from collections import defaultdict

class RecommenderUtils:
    def __init__(self, model, anime_animeEncoded, anime_weights, animeEncoded_anime, user_userEncoded, userEncoded_user, data, ratings_df, user_weights):
        self.model = model
        self.anime_animeEncoded = anime_animeEncoded
        self.anime_weights = anime_weights
        self.animeEncoded_anime = animeEncoded_anime
        self.user_userEncoded = user_userEncoded
        self.userEncoded_user = userEncoded_user
        self.data = data
        self.ratings_df = ratings_df
        self.user_weights = user_weights

    def getAnimeFrame(self, anime):
        if isinstance(anime, int):
            return self.data[self.data.animeID == anime]
        elif isinstance(anime, str):
            return self.data[self.data.title == anime]
        else:
            return pd.DataFrame()

    def find_similar_animes(self, name, n=10, return_dist=False, neg=False):
        try:
            anime_frame = self.getAnimeFrame(name)
            if anime_frame.empty:
                raise ValueError("Anime not found")

            index = anime_frame.animeID.values[0]
            encoded_index = self.anime_animeEncoded.get(index)
            weights = self.anime_weights
            dists = np.dot(weights, weights[encoded_index])
            sorted_dists = np.argsort(dists)

            n = n + 1

            if neg:
                closest = sorted_dists[:n]
            else:
                closest = sorted_dists[-n:]

            print(f'Animes closest to {name}')

            if return_dist:
                return dists, closest

            SimilarityArr = []

            for close in closest:
                decoded_id = self.animeEncoded_anime.get(close)
                anime_frame = self.getAnimeFrame(decoded_id)
                if anime_frame.empty:
                    continue

                anime_name = anime_frame.title.values[0]
                genre = anime_frame.genres.values[0]
                score = anime_frame.score.values[0]
                episodes = anime_frame.episodes.values[0]
                similarity = dists[close]

                SimilarityArr.append({
                    "anime_id": decoded_id,
                    "name": anime_name,
                    "similarity": similarity,
                    "genre": genre,
                    "episodes": episodes,
                    "score": score
                })

            Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
            return Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)

        except Exception as e:
            print(f'{name}!, Not Found in Anime list')
            print(e)
            return pd.DataFrame()

    def find_similar_users(self, item_input, n=10, return_dist=False, neg=False):
        try:
            encoded_index = self.user_userEncoded.get(item_input)
            if encoded_index is None:
                raise ValueError("User not found")

            weights = self.user_weights
            dists = np.dot(weights, weights[encoded_index])
            sorted_dists = np.argsort(dists)

            n = n + 1

            if neg:
                closest = sorted_dists[:n]
            else:
                closest = sorted_dists[-n:]

            print(f'> similar to #{item_input}')

            if return_dist:
                return dists, closest

            SimilarityArr = []

            for close in closest:
                similarity = dists[close]
                decoded_id = self.userEncoded_user.get(close)
                if decoded_id is not None:
                    SimilarityArr.append({"similar_users": decoded_id, "similarity": similarity})

            Frame = pd.DataFrame(SimilarityArr).sort_values(by="similarity", ascending=False)
            return Frame

        except Exception as e:
            print(f'{item_input}!, Not Found in User list')
            print(e)
            return pd.DataFrame()

    def getFavGenre(self, frame):
        frame = frame.dropna()
        all_genres = defaultdict(int)

        for genre in frame['genre']:
            if isinstance(genre, str):
                for g in genre.split(','):
                    all_genres[g.strip()] += 1

        return list(all_genres.keys())

    def get_user_preferences(self, user_id, verbose=0):
        animes_watched_by_user = self.ratings_df[self.ratings_df.userID == int(user_id)]
        if animes_watched_by_user.empty:
            return pd.DataFrame()

        user_rating_percentile = np.percentile(animes_watched_by_user.rating, 75)
        animes_watched_by_user = animes_watched_by_user[animes_watched_by_user.rating >= user_rating_percentile]
        top_animes_user = (
            animes_watched_by_user.sort_values(by="rating", ascending=False)
            .animeID.values
        )

        anime_df_rows = self.data[self.data["animeID"].isin(top_animes_user)]
        anime_df_rows = anime_df_rows[["title"]]

        if verbose:
            print(f"> User #{user_id} has rated {len(animes_watched_by_user)} movies (avg. rating = {animes_watched_by_user['rating'].mean():.1f})")
            print('> preferred genres')

        return anime_df_rows