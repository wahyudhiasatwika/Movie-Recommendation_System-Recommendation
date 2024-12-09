# -*- coding: utf-8 -*-
"""DBS Foundation Camp_Submission 2_Wahyu Dhia Satwika

# Submission 2 - Movie Recommendation

# Import Library
"""

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from google.colab import files
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

"""# Data Collection"""

!pip install -q kaggle

!kaggle datasets download -d parasharmanas/movie-recommendation-system

"""# Data Understanding

Pada tahap ini dilakukan untuk memahami dataset secara keseluruhan sebelum dilakukan pre-processing dan pemodelan

## Data Loading

Tahap ini dilakukan untuk memuat dataset agar lebih mudah digunakan.
"""

import zipfile

with zipfile.ZipFile('/content/movie-recommendation-system.zip', 'r') as zip_ref:
    zip_ref.extractall('/content')

data_movies = pd.read_csv('/content/movies.csv')
data_rating = pd.read_csv('/content/ratings.csv')

data_movies.head()

data_rating.head()

"""# Exploratory Data Analysis

## Checking Dataset
"""

data_movies.info()

data_movies.describe()

data_movies.shape

"""Untuk data pada dataset movies didapatkan hasil bahwa dataset memiliki data berjumlah 62423 dan 3 kolom."""

data_rating.info()

data_rating.describe()

data_rating.shape

"""Untuk data pada dataset ratings didapatkan hasil bahwa dataset memiliki data berjumlah 25000095 dan 4 kolom. Namun karena resource yang saya miliki tidak begitu besar, maka data yang akan dipakai hanya 50000."""

data_ratings = data_rating.sample(n=50000, random_state=42)

data_ratings

"""Dikarenakan resource yang terbatas, maka hanya digunakan 50000 data.

## Check Missing Value
"""

data_movies.duplicated().sum()

data_movies.isnull().sum()

data_ratings.duplicated().sum()

data_ratings.isnull().sum()

"""Dikarenakan pada hasil output tidak terdapat missing value dan data yang duplikat. Maka proses bisa dapat dilanjutkan ke visualisasi.

## EDA - Univariate Visualization
"""

plt.figure(figsize=(10, 6))
sns.histplot(data_ratings['rating'], bins=5, kde=True, color='skyblue', stat="density")
plt.title("Univariate Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Density")
plt.show()

"""Pada visualisasi untuk Distribusi dari Rating Movie di dalam data_ratings, dapat dilihat bahwa rating 4 paling sering muncul dengan Density lebih dari 0.4. Kemudian dilanjutkan kepada rating 3 dan 5 yang memiliki density 0.3. Sehingga dapat disimpulkan bahwa pengguna paling sering memberikan rating 4."""

# Count of movies by genre
genre_count = data_movies['genres'].str.split('|', expand=True).stack().value_counts()

plt.figure(figsize=(12, 7))
genre_count.plot(kind='bar', color='purple')
plt.title("Univariate Genre Count")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.xticks(rotation=90)
plt.show()

"""Untuk distribusi movie berdasarkan genre dapat dilihat bahwa genre Drama memiliki peminat yang paling tinggi yaitu 25000 atau 50% dari keseluruhan data. Setelah itu disusul oleh genre comedy yang memiliki peminat 17000. Untuk sisa genre lainnya memiliki peminat yang kurang lebih sama yaitu mulai dari 5000 hingga 10000.

## EDA - Multivariate Visualization
"""

genre_expanded = data_movies['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True)

movie_genres = genre_expanded.to_frame('genre')
movie_genres['movieId'] = movie_genres.index

genre_ratings = pd.merge(movie_genres, data_ratings, on='movieId')

genre_avg_rating = genre_ratings.groupby('genre')['rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(x=genre_avg_rating.index, y=genre_avg_rating.values, palette='viridis')
plt.title("Average Rating for Each Genre")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.xticks(rotation=90)
plt.show()

"""Untuk distribusi genre berdasarkan rating menunjukkan bahwa hampir semua genre memiliki rating 3.5 namun dapat dilihat untuk Genre Film-Noir memiliki rating yang paling tinggi yaitu 3.7 disusul oleh genre weestern dan IMAX yang memiliki rating 3.6. Untuk sisanya memiliki rating yang hampir sama yaitu 3.5.

# Data Preparation
 Pada tahap ini dilakukan untuk mempersiapkan data sebelum dilakukan pemodelan. Pada proyek kali ini, akan digunakan Content Based Filtering dan Collaborative Filtering. Sebelum masuk ke dalam preparation untuk setiap filtering, maka perlu dicek terlebih dahulu data yang akan digunakan.
"""

unique_genres = data_movies['genres'].str.split('|', expand=True).stack().unique()
print(unique_genres)

"""Dikarenakan terdapat data (no genre listed), maka lebih baik dihapus karena akan mengganggu saat pemodelan nanti."""

# Menghapus genre "(no genres listed)" dari data_movies
data_movies_cleaned = data_movies[~data_movies['genres'].str.contains('(no genres listed)', na=False)]

print(data_movies_cleaned['genres'].str.split('|', expand=True).stack().unique())

"""## Content-Based Filtering

Untuk content-based filtering, akan digunakan berdasarkan movieId, judul film (title), dan genre.

**Mengubah data menjadi bentuk list**
"""

# Pada tahap ini dilakukan pengonversian menjadi bentuk list
movieId = data_movies["movieId"].tolist()
title = data_movies["title"].tolist()
genres = data_movies["genres"].tolist()

# Menampilkan banyak data dari masing-masing list
print(len(movieId))
print(len(title))
print(len(genres))

"""**Membuat DataFrame baru**

Pada DataFrame ini dibuat
"""

content_based = pd.DataFrame({
    "movieId": movieId,
    "title": title,
    "genres": genres
})
content_based

"""### TF-IDF

Dikarenakan komputer hanya bisa memroses berupa numerik maka diperlukan mengubah data dari string menjadi numerik dengen TFidfVectorizer. Dilakukan konversi untuk kolom genres agar dapat diproses untuk pemodelan nanti.
"""

tfidf = TfidfVectorizer()

tfidf.fit(content_based["genres"])
tfidf.get_feature_names_out()

"""Dilakukan transformasi ke dalam bentuk matriks"""

tfidf_matrix = tfidf.fit_transform(content_based["genres"])

tfidf_matrix.shape

"""Mengubah vektor tf-idf ke dalam bentuk matriks dengan fungsi todense()"""

tfidf_matrix.todense()

pd.DataFrame(
    tfidf_matrix.todense(),
    columns = tfidf.get_feature_names_out(),
    index = content_based.title
)

"""## Collaborative Filtering

Untuk Collaborative Filtering akan digunakan userId, movieId, dan Rating.

Dilakukan encode untuk user id agar menjadi bentuk list.
"""

user_id = data_ratings['userId'].unique().tolist()
print('list user_id: ', user_id)

user_to_user_encoded = {x: i for i, x in enumerate(user_id)}
print('encoded user_id: ', user_to_user_encoded)

user_encoded_to_user = {i: x for i, x in enumerate(user_id)}
print('encoded number to used_id: ', user_encoded_to_user)

"""Dilakukan encode untuk movie id agar menjadi bentuk list."""

movie_id = data_movies['movieId'].unique().tolist()
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_id)}
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_id)}

data_ratings['movie'] = data_ratings['movieId'].map(movie_to_movie_encoded)
data_ratings['user'] = data_ratings['userId'].map(user_to_user_encoded)

num_users = len(user_to_user_encoded)
print(num_users)

num_movies = len(movie_encoded_to_movie)
print(num_movies)

data_ratings['rating'] = data_ratings['rating'].values.astype(np.float32)

min_rating = min(data_ratings['rating'])
max_rating = max(data_ratings['rating'])

print('Number of User: {}, Number of anime: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movies, min_rating, max_rating
))

"""Dapat dilihat jumlah user yaitu 34408 dan jumlah movie 62423. Untuk rating terkecil dapat dilihat yaitu 0.5 dan rating terbesar adalah 5"""

data_ratings

"""Dataset diacak agar saat dilakukan pelatihan, model dapat belajar dengan lebih baik."""

collaborative_based = data_ratings[["user", "movie", "rating"]].sample(frac = 1, random_state = 42)
collaborative_based

"""Data dibagi menjadi 80% train dan 20% test"""

x = data_ratings[["user", "movie"]].values

y = data_ratings['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * data_ratings.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""# Modelling

## Content-Based Filtering - Cosine Similarity

Dalam sistem rekomendasi content-based, cosine similarity digunakan untuk mengukur seberapa mirip dua item berdasarkan fitur konten dimana pada model ini digunakan movieId, title, dan genre. Sistem ini mencoba memberikan rekomendasi kepada pengguna berdasarkan movie-movie yang mirip dengan movie yang telah mereka tonton sebelumnya.

**Kelebihan:**

- Cosine similarity adalah metode yang sangat sederhana untuk mengukur kedekatan antar item berdasarkan fitur yang ada dan mudah diimplementasikan.

- Dengan menggunakan cosine similarity, sistem dapat memberikan rekomendasi berdasarkan kesamaan antara item-item yang memiliki karakteristik yang mirip, tanpa memerlukan data interaksi pengguna.

- Sistem ini tidak bergantung pada perilaku pengguna sebelumnya.

- Menghasilkan rekomendasi yang lebih relevan untuk pengguna berdasarkan kesamaan konten yang mereka tonton di masa lalu.

**Kekurangan:**

- Cosine similarity hanya melihat kesamaan konten antar item, tanpa mempertimbangkan preferensi spesifik pengguna.

- Jika pengguna memiliki preferensi yang sangat bervariasi atau sulit diprediksi hanya dari konten, cosine similarity bisa jadi terbatas.

- Hasil dari cosine similarity sangat bergantung pada pemilihan fitur yang baik. Jika fitur yang digunakan untuk representasi item tidak cukup informatif atau kurang tepat, kualitas rekomendasinya akan menurun.
"""

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

"""cosine_similarity digunakan untuk menghitung kesamaan antar item berdasarkan representasi TF-IDF (Term Frequency-Inverse Document Frequency) yang telah dihitung sebelumnya."""

cosine_sim_df = pd.DataFrame(cosine_sim, index=data_movies['title'], columns=data_movies['title'])
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""# Pengujian Model - Cosine Similarity"""

def movie_recommendations(title_movie, similarity_data=cosine_sim_df, items=data_movies[['title', 'genres']], k=10):

    index = similarity_data.loc[:,title_movie].to_numpy().argpartition(
        range(-1, -k, -1))

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop title_movie
    closest = closest.drop(title_movie, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

movie_recommendations("Jumanji (1995)")

"""Dapat dilihat fungsi dapat ditampilkan untuk movie yang memiliki kemiripan genre yang sama yaitu Adventure, Children, dan Fantasy.

## Collaborative Filtering - RecommenderNet

RecommenderNet merupakan sebuah pendekatan dalam sistem rekomendasi yang menggabungkan konsep collaborative filtering dengan deep learning. Biasanya, model ini menggunakan neural networks untuk mengembangkan model rekomendasi berbasis interaksi pengguna-item, dan lebih fokus pada mempelajari pola preferensi pengguna yang tidak dapat diekspresikan dengan mudah menggunakan teknik tradisional. Collaborative filtering berfokus pada memprediksi item yang mungkin disukai pengguna berdasarkan preferensi atau interaksi pengguna lain yang mirip.

**Kelebihan :**
- Menghasilkan rekomendasi yang lebih personal dan relevan sehingga memungkinkan model untuk mengakomodasi preferensi pengguna yang lebih kompleks dan dinamis.

- Model dapat menemukan pola yang tidak terlihat dengan menggunakan fitur konten saja sehingga menghasilkan rekomendasi yang lebih baik karena melibatkan interaksi pengguna.

- Model ini dapat skala untuk dataset yang besar karena melibatkan neural networks

**Kekurangan :**
- RecommenderNet rentan terhadap masalah overfitting jika data interaksi yang tersedia terbatas, terutama jika dataset tidak cukup besar untuk melatih model secara efektif.

- Memerlukan sumber daya komputasi yang lebih besar dibandingkan dengan metode seperti cosine similarity.

- Kualitas dan akurasi rekomendasi sangat bergantung pada kualitas dan jumlah data interaksi yang tersedia.
"""

class RecommenderNet(tf.keras.Model):

  def __init__(self, num_users, num_movies, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movies = num_movies
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.movie_embedding = layers.Embedding( # layer embeddings movie
        num_movies,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.movie_bias = layers.Embedding(num_movies, 1) # layer embedding movie bias

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # layer embedding 2
    movie_vector = self.movie_embedding(inputs[:, 1]) # layer embedding 3
    movie_bias = self.movie_bias(inputs[:, 1]) # layer embedding 4

    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)

    x = dot_user_movie + user_bias + movie_bias

    return tf.nn.sigmoid(x) # activation sigmoid

model = RecommenderNet(num_users, num_movies, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""Setelah dibuat model, selanjutnya ditentukan loss yang digunakan yaitu BinaryCrossentropy, optimizer Adam, dan metrics RMSE. Selanjutnya dilakukan training dengan batch_size 32 dan epochs 20."""

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 32,
    epochs = 20,
    validation_data = (x_val, y_val)
)

"""Didapatkan hasil sebagai berikut:
- RMSE : 0.1727
- val-loss: 0.6456
- val_RMSE : 0.2481
"""

movie_df = data_movies
rating_df = data_ratings

# Mengambil sample user
user_id = rating_df.userId.sample(1).iloc[0]
movie_visited_by_user = rating_df[rating_df.userId == user_id]

# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html
movie_not_visited = movie_df[~movie_df['movieId'].isin(movie_visited_by_user.movieId.values)]['movieId']
movie_not_visited = list(
    set(movie_not_visited)
    .intersection(set(movie_to_movie_encoded.keys()))
)

movie_not_visited = [[movie_to_movie_encoded.get(x)] for x in movie_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_visited), movie_not_visited)
)

"""# Pengujian Model"""

ratings = model.predict(user_movie_array).flatten()

top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_visited[x][0]) for x in top_ratings_indices
]

print('Menampilkan rekomendasi untuk pengguna: {}'.format(user_id))
print('===' * 15)
print('movie dengan rating tertinggi dari pengguna')
print('---' * 15)

top_movie_user = (
    movie_visited_by_user.sort_values(
        by='rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)

movie_df_rows = movie_df[movie_df['movieId'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ':', row.genres)

print('---' * 15)
print('Rekomendasi 10 movie teratas')
print('---' * 15)

recommended_movie = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.title, ':', row.genres)

"""Dapat dilihat hasil 10 rekomendasi movie teratas berdsarkan interaksi pengguna dan rating yang diberikan.

# Evaluation
"""

# Membuat line plot untuk menunjukkan metrik evaluasi
plt.plot(history.history["root_mean_squared_error"])
plt.plot(history.history["val_root_mean_squared_error"])

# Menambahkan judul, label, dan legend pada plot
plt.title("Metrik Evaluasi pada Model")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.xlabel("Epoch")
plt.legend(["Training", "Validation"], loc = "upper right")

# Menampilkan plot
plt.show()

"""Pada hasil evaluasi dapat dilihat untuk akurasi training rmse terus menurun hingga epoch 20. Namun untuk validation rmse, penurunan terhenti pada epoch 10 dan terjadi naik turun untuk hasilnya."""