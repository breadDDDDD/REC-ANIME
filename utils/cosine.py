from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class UserEmbeddingMatcher:
    def __init__(self, anime_animeEncoded, anime_weights, user_weights, userEncoded_user, test_df):
        self.anime_animeEncoded = anime_animeEncoded
        self.anime_weights = anime_weights
        self.user_weights = user_weights
        self.userEncoded_user = userEncoded_user
        self.test_df = test_df

    def embed_new_user(self, user_ratings):
        embeddings = []
        ratings = []

        for anime_id, rating in user_ratings:
            encoded = self.anime_animeEncoded.get(anime_id)
            if encoded is not None:
                embeddings.append(self.anime_weights[encoded])
                ratings.append(rating)

        if embeddings:
            user_emb = np.average(embeddings, axis=0, weights=ratings)
            return user_emb
        else:
            return None

    def find_similar_user(self, new_user_emb):
        similarities = cosine_similarity([new_user_emb], self.user_weights)[0]
        most_similar_idx = np.argmax(similarities)
        most_similar_user_id = self.userEncoded_user[most_similar_idx]
        return most_similar_user_id, similarities[most_similar_idx]

    def get_similar_test(self, user_id):
        user_ratings = self.test_df[self.test_df['userID'] == user_id].drop(['userID'], axis=1).values.tolist()
        new_user_emb = self.embed_new_user(user_ratings)

        if new_user_emb is not None:
            most_similar_user_id, similarity_score = self.find_similar_user(new_user_emb)
            print(f"Most similar user: {most_similar_user_id}, similarity: {similarity_score:.4f}")
            return most_similar_user_id, similarity_score
        else:
            print("No match at all")
            return None, None
