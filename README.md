# Movie-Recommendation System - Wahyu Dhia Satwika

# Project Overview
Di era digital yang terus berkembang, platform streaming film semakin populer, memungkinkan pengguna untuk mengakses berbagai pilihan film dari berbagai genre dan negara. Dengan banyaknya pilihan film yang tersedia, seringkali pengguna merasa kesulitan dalam menemukan film yang sesuai dengan preferensi mereka. Di sinilah sistem rekomendasi berperan penting dalam membantu pengguna memilih film yang relevan berdasarkan riwayat tontonan mereka atau preferensi yang ditentukan. Sistem rekomendasi berbasis konten (content-based filtering) dan berbasis kolaborasi (collaborative filtering) dapat memberikan solusi personalisasi yang efektif.

Collaborative filtering dan content-based filtering merupakan dua metodologi yang banyak digunakan dalam sistem rekomendasi. Content-based filtering melibatkan penyaringan konten berdasarkan informasi atau atribut spesifik untuk menghasilkan rekomendasi yang disesuaikan dengan profil pengguna atau interaksinya. Sebaliknya, collaborative filtering fokus pada mengidentifikasi pengguna dengan preferensi serupa dan menghasilkan rekomendasi berdasarkan preferensi pengguna-pengguna yang serupa ini (Armadhani & Wibowo, 2023).

Dengan membangun sistem rekomendasi ini, diharapkan dapat meningkatkan pengalaman pengguna dalam menemukan film-film baru yang sesuai dengan selera mereka dan meningkatkan interaksi dengan platform streaming.

# Business Understanding
## Problem Statements
- Bagaimana cara membuat sistem rekomendasi movie yang merekomendasikan pengguna berdasarkan genre movie?
- Bagaimana cara mengukur performa model sistem rekomendasi?

## Goals
- Menghasilkan rekomendasi movie sebanyak Top-N Rekomendasi kepada pengguna berdasarkan genre movie.
- Performa model sistem rekomendasi diukur menggunakan metriks evaluasi.

## Solution Statements
- Model Sistem Rekomendasi akan dibuatkan menggunakan Content-Based Filtering dengan pendekatan Cosine Similarity dimana nantinya sistem ini akan merekomendasikan movie bedasarkan kesamaan genre movie. 

- Model Sistem Rekomendasi akan menggunakan Collaborative Filtering dengan pendekatan model based Deep learning yaitu RecommenderNet. Pendekatan ini akan memanfaatkan rating dari pengguna untuk memberikan rekomendasi movie.

- Evaluasi Performa Model akan digunakan Root Mean Squared Error untuk memberikan informasi mengenai keakuratan yang akan dihasilkan oleh model.

# Data Understanding
Dataset Movie Recommendation yang berasal dari kaggle merupakan sebuah dataset yang berasal dari MovieLens Dataset. Dataset dapat diliat melalui [(Dataset Movie Recommendation)](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system). Dataset Movie Recommendation memiliki 2 files yaitu movies.csv yang berjumlah 62423 dan rating.csv yang berjumlah 25000095, dengan rincian sebagai berikut:

**Informasi Dataset**
# Loan Approval Classification Dataset

| **Judul**       | Movie Recommendation System                                                        |                  
|-----------------|-------------------------------------------------------------------------------------|
| **Author**      | MANAS PARASHAR                                                                            |
| **Source**      | [Kaggle](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system) |
| **Visibility**  | Public                                                                              |
| **Usability**   | 10.00                                                                               |

**Metadata**
# Merged Dataset Information

The following table provides an overview of the merged dataset, including column names, data types, and explanations for each column.

| Column    | Dtype   | Deskripsi                                                             |
|-----------|---------|----------------------------------------------------------------------------|
| movieId   | int64   | ID untuk setiap movie |
| title     | object  | Judul Movie                           |
| genres    | object  | Jenis Movie                           |
| userId    | int64   | ID untuk setiap user             |
| rating    | float64 | Penilaian user terhadap movie     |
| timestamp | int64   | Waktu kapan user memberikan penilaian |


# Exploratory Data Analysis
Di dalam dataset terdapat 2 file yaitu movies.csv dan ratings.csv sebagai berikut:

**movies.csv**

| Column   | Non-Null Count | Dtype  |
|----------|----------------|--------|
| movieId  | 62423 non-null | int64  |
| title    | 62423 non-null | object |
| genres   | 62423 non-null | object |

Output di atas menunjukkan bahwa file movies.csv memiliki 62423 data dan 3 kolom. Data tersebut disimpan kedalam variabel data_movie.

**ratings.csv**

| Column    | Non-Null Count | Dtype   |
|-----------|----------------|---------|
| userId    | 25,000,095     | int64   |
| movieId   | 25,000,095     | int64   |
| rating    | 25,000,095     | float64 |
| timestamp | 25,000,095     | int64   |

Output di atas menunjukkan bahwa file ratings.csv memiliki 25000095 data dan 3 kolom. Namun karena resource yang saya miliki tidak begitu besar, maka data yang akan dipakai hanya 50000. Data tersebut disimpan kedalam variabel data_ratings.

**Statistics**

| Column    | count       | mean        | std          | min          | 25%          | 50%          | 75%          | max          |
|-----------|-------------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|
| userId    | 25,000,010  | 81,189.28   | 46,791.72    | 1            | 40,510       | 80,914       | 121,557      | 162,541      |
| movieId   | 25,000,010  | 21,387.98   | 39,198.86    | 1            | 1,196        | 2,947        | 8,623        | 209,171      |
| rating    | 25,000,010  | 3.53        | 1.06         | 0.5          | 3            | 3.5          | 4            | 5            |
| timestamp | 25,000,010  | 1,215,601,000| 226,875,800  | 789,652,000  | 1,011,747,000| 1,198,868,000| 1,447,205,000| 1,574,328,000|

Digunakan describe() untuk memberikan informasi statistik.

- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum.
- 25% adalah kuartil pertama.
- 50% adalah kuartil kedua.
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

## Check Missing Value
**data_movies**
| Column   | Duplicate Count | Missing Values Count |
|----------|-----------------|----------------------|
| movieId  | 0               | 0                    |
| title    | 0               | 0                    |
| genres   | 0               | 0                    |

**data_ratings**
| Column    | Duplicate Count | Missing Values Count |
|-----------|-----------------|----------------------|
| userId    | 0               | 0                    |
| movieId   | 0               | 0                    |
| rating    | 0               | 0                    |
| timestamp | 0               | 0                    |

Dikarenakan pada hasil output tidak terdapat missing value dan data yang duplikat. Maka proses bisa dapat dilanjutkan ke visualisasi.

# EDA - Univariate Visualization
![alt text](https://github.com/wahyudhiasatwika/Movie-Recommendation_System-Recommendation/blob/main/Gambar/uni1.png?raw=true)

Pada visualisasi untuk Distribusi dari Rating Movie di dalam data_ratings, dapat dilihat bahwa rating 4 paling sering muncul dengan Density lebih dari 0.4. Kemudian dilanjutkan kepada rating 3 dan 5 yang memiliki density 0.3. Sehingga dapat disimpulkan bahwa pengguna paling sering memberikan rating 4.

![alt text](https://github.com/wahyudhiasatwika/Movie-Recommendation_System-Recommendation/blob/main/Gambar/uni2.png?raw=true)

Untuk distribusi movie berdasarkan genre dapat dilihat bahwa genre Drama memiliki peminat yang paling tinggi yaitu 25000 atau 50% dari keseluruhan data. Setelah itu disusul oleh genre comedy yang memiliki peminat 17000. Untuk sisa genre lainnya memiliki peminat yang kurang lebih sama yaitu mulai dari 5000 hingga 10000.

# EDA - Multivariate
![alt text](https://github.com/wahyudhiasatwika/Movie-Recommendation_System-Recommendation/blob/main/Gambar/multi1.png?raw=true)

Untuk distribusi genre berdasarkan rating menunjukkan bahwa hampir semua genre memiliki rating 3.5 namun dapat dilihat untuk Genre Film-Noir memiliki rating yang paling tinggi yaitu 3.7 disusul oleh genre weestern dan IMAX yang memiliki rating 3.6. Untuk sisanya memiliki rating yang hampir sama yaitu 3.5.

# Data Preparation 
Pada tahap ini dilakukan untuk mempersiapkan data sebelum dilakukan pemodelan. Pada proyek kali ini, akan digunakan Content Based Filtering dan Collaborative Filtering. Sebelum masuk ke dalam preparation untuk setiap filtering, maka perlu dicek terlebih dahulu data yang akan digunakan. 

Dikarenakan terdapat data (no genre listed), maka lebih baik dihapus karena akan mengganggu saat pemodelan nanti sebagai berikut.

['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama'
 'Action', 'Crime', 'Thriller', 'Horror' 'Mystery', 'Sci-Fi', 'IMAX'
 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir']

## Content-Based Filtering
Content-based filtering adalah salah satu metode dalam sistem rekomendasi yang digunakan untuk memberikan rekomendasi berdasarkan informasi atau atribut yang terkandung dalam item yang dianalisis. Dalam konteks sistem rekomendasi film ini, content-based filtering berfokus pada fitur film itu sendiri, seperti `movieID`, `title`, dan `genre`.

Pertama-tama, data akan dilakukan konversi menjadi bentuk list kemudian akan disimpan menjadi dataframe baru yaitu `content_based`. Setelah itu akan diproses menggunakan TF-IDF. 

### TF-IDF 
Dikarenakan komputer hanya bisa memroses berupa numerik maka diperlukan mengubah data dari string menjadi numerik dengen TFidfVectorizer. Dilakukan konversi untuk kolom genres agar dapat diproses untuk pemodelan nanti.

Setelah itu dilakukan transformasi ke dalam bentuk matrix dan mengubah vektor tf-idf ke dalam bentuk matriks dengan fungsi todense(). Sehingga contoh dataframe yang baru sebagai berikut:

| title                                | action  | adventure | animation | children | comedy  | crime  | documentary | drama  | fantasy  | fi     | ... | musical | mystery | no  | noir  | romance | sci    | thriller | war    | western |
|--------------------------------------|---------|-----------|-----------|----------|---------|--------|-------------|--------|----------|--------|-----|---------|---------|-----|-------|---------|--------|----------|--------|---------|
| Toy Story (1995)                     | 0.000000| 0.446566  | 0.48833   | 0.488084 | 0.277717| 0.0    | 0.0         | 0.000000| 0.496748 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.000000 | 0.0    | 0.0     |
| Jumanji (1995)                       | 0.000000| 0.539795  | 0.00000   | 0.589981 | 0.000000| 0.0    | 0.0         | 0.000000| 0.600454 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.000000 | 0.0    | 0.0     |
| Grumpier Old Men (1995)              | 0.000000| 0.000000  | 0.00000   | 0.000000 | 0.598464| 0.0    | 0.0         | 0.000000| 0.000000 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.801149 | 0.0    | 0.0     |
| Waiting to Exhale (1995)             | 0.000000| 0.000000  | 0.00000   | 0.000000 | 0.537355| 0.0    | 0.0         | 0.440220| 0.000000 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.719344 | 0.0    | 0.0     |
| Father of the Bride Part II (1995)   | 0.000000| 0.000000  | 0.00000   | 0.000000 | 1.000000| 0.0    | 0.0         | 0.000000| 0.000000 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.000000 | 0.0    | 0.0     |
| We (2018)                            | 0.000000| 0.000000  | 0.00000   | 0.000000 | 0.000000| 0.0    | 0.0         | 1.000000| 0.000000 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.000000 | 0.0    | 0.0     |
| Window of the Soul (2001)            | 0.000000| 0.000000  | 0.00000   | 0.000000 | 0.000000| 0.0    | 1.0         | 0.000000| 0.000000 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.000000 | 0.0    | 0.0     |
| Bad Poems (2018)                     | 0.000000| 0.000000  | 0.00000   | 0.000000 | 0.773558| 0.0    | 0.0         | 0.633726| 0.000000 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.000000 | 0.0    | 0.0     |
| A Girl Thing (2001)                  | 0.000000| 0.000000  | 0.00000   | 0.000000 | 0.000000| 0.0    | 0.0         | 0.000000| 0.000000 | 0.0    | ... | 0.57735 | 0.0     | 0.0 | 0.57735| 0.0    | 0.000000 | 0.0    | 0.0     |
| Women of Devil's Island (1962)       | 0.601845| 0.711583  | 0.00000   | 0.000000 | 0.000000| 0.0    | 0.0         | 0.362536| 0.000000 | 0.0    | ... | 0.00000 | 0.0     | 0.0 | 0.0   | 0.0     | 0.0    | 0.000000 | 0.0    | 0.0     |

## Collaborative Filtering
Collaborative Filtering adalah salah satu pendekatan dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan pola interaksi atau preferensi dari pengguna lain. Pada projek ini digunakan `movieID`, `userID`, dan `rating`. 

Setelah dilakukan encoding, maka dataframe akan menjadi sebagai berikut:

| userId  | movieId | rating | timestamp  | movie | user |
|---------|---------|--------|------------|-------|------|
| 15347762| 99476   | 3.5    | 1467897440 | 20150 | 0    |
| 16647840| 107979  | 4.0    | 994007728  | 2543  | 1    |
| 23915192| 155372  | 3.0    | 1097887531 | 1556  | 2    |
| 10052313| 65225   | 4.0    | 1201382275 | 7028  | 3    |
| 12214125| 79161   | 5.0    | 1488915363 | 495   | 4    |
| ...     | ...     | ...    | ...        | ...   | ...  |
| 18674093| 120998  | 3.5    | 1093971335 | 7884  | 3173 |
| 19535136| 126848  | 5.0    | 1520322201 | 36842 | 34405|
| 18309526| 118603  | 1.0    | 952879431  | 2618  | 34406|
| 3444719 | 22712   | 4.0    | 1529264095 | 1879  | 34407|
| 9860932 | 64007   | 5.0    | 913411726  | 1992  | 31578|

**Penjelasan:**
- **userId**: ID unik user.
- **movieId**: ID unik movie.
- **rating**: Penilaian yang diberikan oleh user.
- **timestamp**: Waktu ketika rating diberikan.
- **movie**: ID terkait film .
- **user**: ID pengguna yang memberikan rating.

# Modelling & Result

##  Content-Based Filtering - Cosine Similarity

Dalam sistem rekomendasi content-based, cosine similarity digunakan untuk mengukur seberapa mirip dua item berdasarkan fitur konten dimana pada model ini digunakan movieId, title, dan genre. Sistem ini mencoba memberikan rekomendasi kepada pengguna berdasarkan movie-movie yang mirip dengan movie yang telah mereka tonton sebelumnya.

Kelebihan:

Cosine similarity adalah metode yang sangat sederhana untuk mengukur kedekatan antar item berdasarkan fitur yang ada dan mudah diimplementasikan.

Dengan menggunakan cosine similarity, sistem dapat memberikan rekomendasi berdasarkan kesamaan antara item-item yang memiliki karakteristik yang mirip, tanpa memerlukan data interaksi pengguna.

Sistem ini tidak bergantung pada perilaku pengguna sebelumnya.

Menghasilkan rekomendasi yang lebih relevan untuk pengguna berdasarkan kesamaan konten yang mereka tonton di masa lalu.

Kekurangan:

Cosine similarity hanya melihat kesamaan konten antar item, tanpa mempertimbangkan preferensi spesifik pengguna.

Jika pengguna memiliki preferensi yang sangat bervariasi atau sulit diprediksi hanya dari konten, cosine similarity bisa jadi terbatas.

Hasil dari cosine similarity sangat bergantung pada pemilihan fitur yang baik. Jika fitur yang digunakan untuk representasi item tidak cukup informatif atau kurang tepat, kualitas rekomendasinya akan menurun.

`cosine_similarity()` digunakan untuk menghitung kesamaan antar item berdasarkan representasi TF-IDF (Term Frequency-Inverse Document Frequency) yang telah dihitung sebelumnya. Setelah digunakan fungsi tersebut, maka dihasilkan dataframe seperti contoh sebagai berikut. 

| title                                                       | Welcome Mr. Marshall (Bienvenido Mister Marshall) (1953) | Mine Games (2012) | The Night of the Grizzly (1966) | The Pirates of Penzance (1985) | Shikhar (2005) |
|-------------------------------------------------------------|----------------------------------------------------------|-------------------|-------------------------------|--------------------------------|----------------|
| Scooby-Doo! Frankencreepy (2014)                             | 0.000000                                                 | 0.411958          | 0.285718                      | 0.443549                       | 0.0            |
| Don't Be a Sucker! (1947)                                    | 0.000000                                                 | 0.000000          | 0.000000                      | 0.000000                       | 0.0            |
| Armwrestler From Solitude, The (Armbryterskan från Ensamheten) (2004) | 0.000000                                                 | 0.000000          | 0.000000                      | 0.000000                       | 0.0            |
| Mala Mala (2014)                                             | 0.000000                                                 | 0.000000          | 0.000000                      | 0.000000                       | 0.0            |
| Albino Alligator (1996)                                      | 0.000000                                                 | 0.385302          | 0.000000                      | 0.000000                       | 0.0            |
| Happy Go Ducky (1958)                                        | 0.373180                                                 | 0.000000          | 0.000000                      | 0.754596                       | 0.0            |
| Aquarelle (1958)                                             | 0.773558                                                 | 0.000000          | 0.000000                      | 0.382558                       | 0.0            |
| One-Way Trip to Antibes, A (2011)                            | 0.000000                                                 | 0.000000          | 0.000000                      | 0.000000                       | 0.0            |
| Best Intentions, The (Den goda viljan) (1992)                | 0.000000                                                 | 0.000000          | 0.000000                      | 0.000000                       | 0.0            |
| Breathing (2011)                                            | 0.000000                                                 | 0.000000          | 0.000000                      | 0.000000                       | 0.0            |

**Penjelasan:**
Setiap kolom berikutnya menggambarkan tingkat kemiripan antara film yang bersangkutan dengan film lainnya berdasarkan genre dimana nilai pada setiap cell menunjukkan tingkat kemiripan, yang dihitung berdasarkan perbandingan genre antara dua film. Nilai 0 menunjukkan bahwa tidak ada kemiripan antara genre film tersebut.

### Pengujian Model
Dilakukan pengujian sebagai berikut `movie_recommendations("Jumanji (1995)")`:
| title                                                                                     | genres                          |
|-------------------------------------------------------------------------------------------|---------------------------------|
| Santa Claus: The Movie (1985)                                                              | Adventure|Children|Fantasy     |
| Polar Bear King, The (Kvitebjørn Kong Valemon) (1991)                                      | Adventure|Children|Fantasy     |
| Chronicles of Narnia: The Voyage of the Dawn Treader (2010)                               | Adventure|Children|Fantasy     |
| Little Ghost (1997)                                                                        | Adventure|Children|Fantasy     |
| Percy Jackson: Sea of Monsters (2013)                                                      | Adventure|Children|Fantasy     |
| Borrowers, The (2011)                                                                     | Adventure|Children|Fantasy     |
| The Chronicles of Narnia: The Lion, the Witch and the Wardrobe (2005)                      | Adventure|Children|Fantasy     |
| Pelicanman (Pelikaanimies) (2004)                                                          | Adventure|Children|Fantasy     |
| Chronicles of Narnia: Prince Caspian, The (2008)                                           | Adventure|Children|Fantasy     |
| Old Man Khottabych (1956)                                                                  | Adventure|Children|Fantasy     |

## Collaborative Filtering - RecommenderNet
RecommenderNet merupakan sebuah pendekatan dalam sistem rekomendasi yang menggabungkan konsep collaborative filtering dengan deep learning. Biasanya, model ini menggunakan neural networks untuk mengembangkan model rekomendasi berbasis interaksi pengguna-item, dan lebih fokus pada mempelajari pola preferensi pengguna yang tidak dapat diekspresikan dengan mudah menggunakan teknik tradisional. Collaborative filtering berfokus pada memprediksi item yang mungkin disukai pengguna berdasarkan preferensi atau interaksi pengguna lain yang mirip.

Kelebihan :

Menghasilkan rekomendasi yang lebih personal dan relevan sehingga memungkinkan model untuk mengakomodasi preferensi pengguna yang lebih kompleks dan dinamis.

Model dapat menemukan pola yang tidak terlihat dengan menggunakan fitur konten saja sehingga menghasilkan rekomendasi yang lebih baik karena melibatkan interaksi pengguna.

Model ini dapat skala untuk dataset yang besar karena melibatkan neural networks

Kekurangan :

RecommenderNet rentan terhadap masalah overfitting jika data interaksi yang tersedia terbatas, terutama jika dataset tidak cukup besar untuk melatih model secara efektif.

Memerlukan sumber daya komputasi yang lebih besar dibandingkan dengan metode seperti cosine similarity.

Kualitas dan akurasi rekomendasi sangat bergantung pada kualitas dan jumlah data interaksi yang tersedia.

Model dibuat dengan loss yang digunakan yaitu BinaryCrossentropy, optimizer Adam, dan metrics RMSE. Selanjutnya dilakukan training dengan batch_size 32 dan epochs 20. Output yang dihasilkan yaitu:
- RMSE : 0.1727
- val-loss: 0.6456
- val_RMSE : 0.2481

Kemudian dilakukan pengujian dan dihasilkan output sebagai berikut:

**Rekomendasi Film untuk Pengguna 130921**

**Movie dengan Rating Tertinggi dari Pengguna**
**Oculus (2013)**  
**Genre**: Horror

---

**Rekomendasi 10 Movie Teratas**
1. **The Shawshank Redemption (1994)**  
   **Genre**: Crime|Drama
2. **Wallace & Gromit: A Close Shave (1995)**  
   **Genre**: Animation|Children|Comedy
3. **The Godfather (1972)**  
   **Genre**: Crime|Drama
4. **Some Like It Hot (1959)**  
   **Genre**: Comedy|Crime
5. **Monty Python's Life of Brian (1979)**  
   **Genre**: Comedy
6. **Dial M for Murder (1954)**  
   **Genre**: Crime|Mystery|Thriller
7. **A Streetcar Named Desire (1951)**  
   **Genre**: Drama
8. **The Good, the Bad and the Ugly (1966)**  
   **Genre**: Action|Adventure|Western
9. **The Godfather: Part II (1974)**  
   **Genre**: Crime|Drama
10. **Das Boot (1981)**  
    **Genre**: Action|Drama|War

# Evaluation

![alt text](https://github.com/wahyudhiasatwika/Movie-Recommendation_System-Recommendation/blob/main/Gambar/eval1.png?raw=true)

Pada hasil evaluasi dapat dilihat untuk akurasi training rmse terus menurun hingga epoch 20. Namun untuk validation rmse, penurunan terhenti pada epoch 10 dan terjadi naik turun untuk hasilnya.

# Referensi
Movie Recommendation System Based on Synopsis Using Content-Based Filtering with TF-IDF and Cosine Similarity [_(Sumber Referensi)] (https://doi.org/10.21108/ijoict.v9i2.747)