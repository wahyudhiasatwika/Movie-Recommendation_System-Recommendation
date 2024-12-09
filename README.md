# Movie-Recommendation - Wahyu Dhia Satwika

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

## SOlution Statements
- Model Sistem Rekomendasi akan dibuatkan menggunakan Content-Based Filtering dengan pendekatan Cosine Similarity dimana nantinya sistem ini akan merekomendasikan movie bedasarkan kesamaan genre movie. 

- Model Sistem Rekomendasi akan menggunakan Collaborative Filtering dengan pendekatan model based Deep learning yaitu RecommenderNet. Pendekatan ini akan memanfaatkan rating dari pengguna untuk memberikan rekomendasi movie.

- Evaluasi Performa Model akan digunakan Root Mean Squared Error untuk memberikan informasi mengenai keakuratan yang akan dihasilkan oleh model.

# Data Understanding
Dataset Movie Recommendation yang berasal dari kaggle merupakan sebuah dataset yang berasal dari MovieLens Dataset. Dataset dapat diliat melalui [(Dataset Movie Recommendation)](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system). Dataset Movie Recommendation memiliki 2 files yaitu movies.csv yang berjumlah 62423 dan rating.csv yang berjumlah 25000095, dengan rincian sebagai berikut:

**Informasi Dataset**
# Loan Approval Classification Dataset

| **Judul**       | Movie Recommendation System                                                        |                  
|-----------------|-------------------------------------------------------------------------------------|
| **Author**      | MANAS PARASHAR LO                                                                           |
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


# Referensi
Movie Recommendation System Based on Synopsis Using Content-Based Filtering with TF-IDF and Cosine Similarity [_(Sumber Referensi)] (https://doi.org/10.21108/ijoict.v9i2.747)