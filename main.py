
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD
import warnings; warnings.simplefilter('ignore')




credits = pd.read_csv('./input_data/movie_dataset/credits.csv')
keywords = pd.read_csv('./input_data/movie_dataset/keywords.csv')
links_small = pd.read_csv('./input_data/movie_dataset/links_small.csv')
md = pd.read_csv('./input_data/movie_dataset/movies_metadata.csv')
ratings = pd.read_csv('./input_data/movie_dataset/ratings_small.csv')







def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    id_map = pd.read_csv('./links_small.csv')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
    # id_map = id_map.set_index('tmdbId')
    indices_map = id_map.set_index('id')

    def hybrid(userId, title):
        idx = indices[title]
        tmdbId = id_map.loc[title]['id']
        movie_id = id_map.loc[title]['movieId']
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:26]
        movie_indices = [i[0] for i in sim_scores]
        movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'release_date', 'id']]
        movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
        movies = movies.sort_values('est', ascending=False)
        return movies.head(10)