import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, precision_score

columns = ['userId', 'movieId', 'rating', 'timestamp']
frame = pd.read_csv('ml-latest-small/ratings.csv', sep=',', names=columns)
frame = frame[1:]
frame['rating'] = pd.to_numeric(frame['rating'])

columns = ['movieId', 'movie title', 'genres']
movies = pd.read_csv('ml-latest-small/movies.csv', sep=',', names=columns, encoding='latin-1')
movie_names = movies[['movieId', 'movie title']]
movies = movies[1:]



combined_movies_data = pd.merge(frame, movie_names, on='movieId')

rating_crosstab = combined_movies_data.pivot_table(values='rating', index='userId',fill_value=0, columns='movie title')

X = rating_crosstab.T
SVD = TruncatedSVD(n_components=12, random_state=17)
resultant_matrix = SVD.fit_transform(X)

corr_mat = np.corrcoef(resultant_matrix)

movie_names = rating_crosstab.columns
movies_list = list(movie_names)
forrest_gump = movies_list.index('Forrest Gump (1994)')
corr_star_wars = corr_mat[forrest_gump]

# print(list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.9)]))
most_similar_movies = list(movie_names[(corr_star_wars<1.0) & (corr_star_wars > 0.95)])


movies_seperated_by_id = combined_movies_data.groupby('movie title')

list1=[]
list2=[]

for i in movies_seperated_by_id:
    if(i[0] == 'Forrest Gump (1994)'):
        users = i[1].userId
        ratings = i[1].rating
        ratings_len = len(ratings)
        print(ratings_len)
        for j in range (ratings_len):
            if(ratings.iloc[j] > 3):
                list1.append(users.iloc[j]) 
        

for j in movies_seperated_by_id:
    if(j[0] == most_similar_movies[1]):
        users = j[1].userId
        ratings = j[1].rating
        ratings_len = len(ratings)
        for k in range(ratings_len):
            if(ratings.iloc[k] > 3):
                list2.append(users.iloc[k])


c = 0
for i in list1:
    if i in list2:
        c +=1
total = len(list1) + len(list2) - c
print(c , total , c/total)
