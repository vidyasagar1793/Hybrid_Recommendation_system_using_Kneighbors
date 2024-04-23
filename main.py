from collections import namedtuple

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix



def load_data():
    df_movies = pd.read_csv('movie.csv')
    df_ratings = pd.read_csv('rating.csv')

    
    df_movies.rename(columns={'movieId': 'mID', 'title': 'title', 'genres': 'genres'}, inplace=True)
    df_ratings.rename(columns={'userId': 'uID', 'movieId': 'mID', 'rating': 'rating'}, inplace=True)

    return df_movies, df_ratings


df_movies, df_ratings = load_data()


class RecSys():
    def __init__(self, data):
        self.data = data
        self.allusers = list(self.data.users['uID'])
        self.allmovies = list(self.data.movies['mID'])
        self.genres = list(self.data.movies.columns.drop(['mID', 'title']))
        self.mid2idx = dict(zip(self.data.movies.mID, list(range(len(self.data.movies)))))
        self.uid2idx = dict(zip(self.allusers, list(range(len(self.data.users)))))
        self.Mr = self.rating_matrix()

    def rating_matrix(self):
        ind_movie = [self.mid2idx[x] for x in self.data.ratings.mID]
        ind_user = [self.uid2idx[x] for x in self.data.ratings.uID]
        rating_train = list(self.data.ratings.rating)
        return csr_matrix((rating_train, (ind_user, ind_movie)),
                          shape=(len(self.allusers), len(self.allmovies))).toarray()


class HybridRecSys(RecSys):
    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.Mm = self.calc_movie_feature_matrix()
        self.nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.nn_model.fit(csr_matrix(self.Mm))
        self.sim_ids = None
        self.sim_dis = None

    def calc_movie_feature_matrix(self):
        df_movies_genres = df_movies.drop(columns=['mID', 'title'])
        genres_matrix = df_movies_genres.values
        movie_feature_matrix = np.concatenate([genres_matrix, self.Mr.T], axis=1)
        return movie_feature_matrix

    def kneighbors_similarity(self, movie_idx, k=10):
        similar_movies_distances, similar_movies_indices = self.nn_model.kneighbors(self.Mm[movie_idx].reshape(1, -1),
                                                                                    n_neighbors=k + 1,
                                                                                    return_distance=True)
        sorted_indices = np.argsort(similar_movies_distances[0])
        similar_movies_indices = similar_movies_indices[0][sorted_indices[1:]]
        similar_movies_distances = similar_movies_distances[0][sorted_indices[1:]]
        self.sim_ids = similar_movies_indices
        self.sim_dis = similar_movies_distances


data = namedtuple('Data', ['users', 'movies', 'ratings'])
data.users = pd.DataFrame({'uID': df_ratings['uID'].unique()})
data.movies = df_movies
data.ratings = df_ratings

HRS = HybridRecSys(data)



def main():
    st.title("Movie Recommendation System")

    
    movie_title = st.text_input("Enter a movie title:", "Old School")

    # Recommendation button
    if st.button("Get Recommendations"):
        recommend_movies(movie_title)


# Recommendation function
def recommend_movies(movie_title):
    matching_movies = df_movies[df_movies['title'].str.contains(movie_title, case=False)]

    if not matching_movies.empty:
        movie_ids_to_recommend = matching_movies['mID'].values
        for movie_id_to_recommend in movie_ids_to_recommend:
            movie_title_to_recommend = \
            HRS.data.movies.loc[HRS.data.movies['mID'] == movie_id_to_recommend, 'title'].values[0]
            genres_of_recommend = [genre for genre, value in zip(HRS.genres, HRS.Mm[HRS.mid2idx[movie_id_to_recommend]])
                                   if value == 1]
            movie_idx = HRS.mid2idx.get(movie_id_to_recommend)
            k = 10
            HRS.kneighbors_similarity(movie_idx, k)
            similar_movies_scores = 1 / (1 + HRS.sim_dis)
            similar_movies_indices = HRS.sim_ids
            top_movie_ids = [HRS.allmovies[i] for i in similar_movies_indices]

            st.markdown("------------------------------------------------------------------")
            st.markdown("\nLooked up movie successfully...")
            st.markdown("\nMovie: ")
            st.markdown(f"\n{movie_title_to_recommend} - ({', '.join(genres_of_recommend)})")
            st.markdown("\nRecommended Movies:\n")

            for i, movie_id in enumerate(top_movie_ids):
                movie_title = HRS.data.movies.loc[HRS.data.movies['mID'] == movie_id, 'title'].values[0]
                movie_genres = [genre for genre, value in zip(HRS.genres, HRS.Mm[HRS.mid2idx[movie_id]]) if value == 1]
                st.markdown(
                    f"{movie_title} - ({', '.join(movie_genres)}) - (Similarity Score: {similar_movies_scores[i]:.4f})")

    else:
        st.write('Movie not found in the database')


if __name__ == "__main__":
    main()
