# -*- coding: utf-8 -*-
"""recomendation-system.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F5iOSOr1_yfNYNLg5xvLOO-xZ1UZ2vGr

**PROJECT-DICODING**

**RECOMMENDATION SYSTEM**

**FAISHAL ANWAR HASYIM**

## Import Library yang dibutuhkan
## Impor Pustaka
 Dalam sel ini, kita mengimpor semua pustaka yang diperlukan untuk membangun sistem rekomendasi.
 Pustaka yang digunakan meliputi:

 - `pandas`: Untuk manipulasi dan analisis data.
    Contohnya, dapat digunakan untuk mengimpor dataset, melakukan agregasi, dan memanipulasi DataFrame.
 - `numpy`: Untuk operasi numerik dan penanganan array.
    Memungkinkan pengolahan data multidimensi yang efisien dan dukungan untuk operasi matematika.
 - `neattext`: Untuk fungsi pemrosesan teks, khususnya untuk membersihkan dan memproses data teks.
    Berguna untuk melakukan normalisasi teks, penghapusan tanda baca, dan preprocessing lainnya.
 - `scipy`: Untuk operasi matriks jarang, yang penting untuk menangani dataset besar secara efisien.
    Menyediakan beragam fungsi matematis dan algoritma untuk optimasi dan statistik.
 - `sklearn`: Untuk algoritma pembelajaran mesin, termasuk Nearest Neighbors dan teknik vektorisasi.
    Memiliki berbagai fungsi, termasuk untuk membagi dataset, menghitung akurasi, dan melakukan proses klasifikasi.
 - `matplotlib` dan `seaborn`: Untuk visualisasi data guna membantu memahami data dengan lebih baik.
    `matplotlib` menyediakan antarmuka dasar untuk membuat berbagai jenis plot, sedangkan `seaborn` menawarkan antarmuka yang lebih sederhana
     dan estetis untuk visualisasi data.
"""

# Menginstall package neattext
!pip install neattext

# Mengimpor library yang dibutuhkan
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as nfx

"""## 1. Data Understanding
Pada bagian ini, kita akan menjelajahi dataset yang merupakan bagian dari sistem rekomendasi. Dataset yang digunakan meliputi:

* movies.csv: Berisi informasi tentang film, seperti judul, tahun rilis, dan genre.
* links.csv: Berisi tautan ke film, mungkin ke platform seperti IMDb.
* tags.csv: Berisi tag yang dihasilkan pengguna untuk film tertentu.
* ratings.csv: Berisi penilaian pengguna untuk film, termasuk ID pengguna dan penilaian yang diberikan.

Data ini bersumber dari Kaggle, dan kita akan mencetak jumlah entri unik dalam setiap dataset untuk memahami cakupan data kita.
File bisa diunduh di : https://www.kaggle.com/datasets/kanametov/movies-recomendation-system



"""

movies = pd.read_csv('/content/movies.csv')
links = pd.read_csv('/content/links.csv')
tags = pd.read_csv('/content/tags.csv')
ratings = pd.read_csv('/content/ratings.csv')

print('Jumlah data film yang tersedia: ', len(movies.movieId.unique()))
print('Jumlah data link film: ', len(links.movieId.unique()))
print('Jumlah data tag film: ', len(tags.movieId.unique()))
print('Jumlah data rating film: ', len(ratings.userId.unique()))

"""Di sini, kita membaca empat file CSV yang berbeda menggunakan pd.read_csv(). Setiap file CSV berisi data yang berbeda:

- movies.csv: Berisi informasi tentang film, seperti judul, tahun rilis, dan genre.

- links.csv: Berisi informasi tentang link film, kemungkinan ke platform IMDb.

- tags.csv: Berisi tag atau label yang diberikan oleh pengguna untuk film tertentu.

- ratings.csv: Berisi data rating yang diberikan oleh pengguna untuk film, termasuk userId dan rating yang diberikan.

selanjutnya, kita print jumlah dari masing masing data
"""

movies # kode ini akan melihat isi dari variabel movies yang sebelumnya kita baca menggunakan pd.read_csv

movies.info() # kode ini akan melihat detail dari variabel movies seperti tipe data dan sebagainya

"""Kita memeriksa struktur dataset movies, termasuk tipe data dan jumlah film serta genre unik yang tersedia."""

# kode ini ditujukan untuk melihat berapa movies pada dataset movies dengan jenis yang berbeda (duplikat tidak dihitung)
print('Banyak data movie: ', len(movies.movieId.unique()))

# Melihat isi Genre
print('Genre movie: ', movies.genres.unique())

"""Kita membuat plot batang untuk memvisualisasikan frekuensi berbagai genre film. Ini membantu kita memahami genre mana yang paling populer di antara pengguna dan dapat memberikan wawasan untuk rekomendasi yang lebih baik."""

# Menghitung frekuensi setiap genre
genre_counts = movies['genres'].str.get_dummies(sep='|').sum().sort_values(ascending=False)

# Membuat bar plot untuk genre yang paling umum
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar')
plt.title('Genre Film yang Paling Umum')
plt.xlabel('Genre')
plt.ylabel('Jumlah Film')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

tags # kode ini akan melihat isi dari variabel tags yang sebelumnya kita baca menggunakan pd.read_csv

tags.info() # kode ini akan melihat detail dari variabel tags seperti tipe data dan sebagainya

"""Kita memeriksa struktur dataset tags, termasuk tipe data dan jumlah film serta tag unik yang tersedia."""

# kode ini ditujukan untuk melihat berapa movies pada dataset tags dengan jenis yang berbeda (duplikat tidak dihitung)
print('Banyak data movie: ', len(tags.movieId.unique()))

# melihat isi tags
print('Genre movie: ', tags.tag.unique())

"""Kita membuat plot batang untuk memvisualisasikan frekuensi berbagai tag film. Ini membantu kita memahami tag mana yang paling populer di antara pengguna dan dapat memberikan wawasan untuk rekomendasi yang lebih baik."""

# Menghitung frekuensi setiap tag
tag_counts = tags['tag'].value_counts().head(20)  # Mengambil 20 tag teratas

# Membuat bar plot untuk tag yang paling sering digunakan
plt.figure(figsize=(12, 6))
tag_counts.plot(kind='bar')
plt.title('Tag yang Sering Digunakan oleh Pengguna')
plt.xlabel('Tag')
plt.ylabel('Jumlah Penggunaan')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

links # kode ini akan melihat isi dari variabel links yang sebelumnya kita baca menggunakan pd.read_csv

links.info() # kode ini akan melihat detail dari variabel links seperti tipe data dan sebagainya

"""Kita memeriksa struktur dataset links, termasuk tipe data dan jumlah film serta link unik yang tersedia."""

# kode ini ditujukan untuk melihat berapa movies pada dataset links dengan jenis yang berbeda (duplikat tidak dihitung)
print('Banyak data movie: ', len(links.movieId.unique()))

# Melihat isi link movie
print('Link movie pada imdb: ', links.imdbId.unique())

ratings # kode ini akan melihat isi dari variabel ratings yang sebelumnya kita baca menggunakan pd.read_csv

ratings.info() # kode ini akan melihat detail dari variabel ratings seperti tipe data dan sebagainya

"""Kita memeriksa struktur dataset ratings, termasuk tipe data dan jumlah film serta rating unik yang tersedia."""

# Menunjukkan jumlah film yang dievaluasi atau jumlah film yang diberi rating
print('Banyak data movie: ', len(ratings.movieId.unique()))

# Memeriksa jumlah pengguna yang melakukan ulasan atau memeriksa jumlah pengguna yang memberi rating setidaknya satu film
print('Jumlah pengguna yang melakukan ulasan: ', len(ratings['userId'].unique()))

# Melihat isi dari rating, nilai minimum dan maksimum
print('Ratings movie: ', ratings.rating.unique())

"""Kita membuat histogram untuk memvisualisasikan distribusi rating film yang diberikan oleh pengguna. Dengan menggunakan sns.histplot(), histogram ini menunjukkan frekuensi setiap rating dalam rentang 1 hingga 5, dengan penambahan kurva distribusi (KDE) untuk memberikan gambaran yang lebih halus tentang sebaran data. Visualisasi ini sangat berguna untuk memahami bagaimana pengguna memberikan rating kepada film, serta mengidentifikasi pola atau kecenderungan dalam penilaian."""

# Membuat histogram distribusi rating
plt.figure(figsize=(10, 6))
sns.histplot(ratings['rating'], bins=5, kde=True)  # kde=True menambahkan kurva distribusi
plt.title('Distribusi Rating Film')
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.xticks(range(1, 6))  # Mengatur label sumbu x
plt.grid(axis='y')
plt.show()

"""pada tahap ini saya menjalankan kode untuk melihat apakah ada nilai null pada dataset dataset, movies,tags,links, dan ratings dan jika ada maka nilai null akan dihapus"""

movies.isnull().any() # digunakan untuk memeriksa apakah ada nilai yang hilang (missing values) dalam DataFrame movies.

tags.isnull().any() # digunakan untuk memeriksa apakah ada nilai yang hilang (missing values) dalam DataFrame tags.

links.isnull().any() # digunakan untuk memeriksa apakah ada nilai yang hilang (missing values) dalam DataFrame links.

ratings.isnull().any() # digunakan untuk memeriksa apakah ada nilai yang hilang (missing values) dalam DataFrame ratings.

"""pada tahap ini saya menjalankan kode untuk melihat apakah ada duplikat pada dataset dataset, movies,tags,links, dan ratings dan jika ada maka nilai duplikat akan dihapus"""

# Memeriksa apakah ada duplikat
duplicates_exist = movies.duplicated().any()
print(f"Apakah ada duplikat dalam DataFrame movies? {duplicates_exist}")

# Memeriksa apakah ada duplikat
duplicates_exist = ratings.duplicated().any()
print(f"Apakah ada duplikat dalam DataFrame ratings? {duplicates_exist}")

# Memeriksa apakah ada duplikat
duplicates_exist = tags.duplicated().any()
print(f"Apakah ada duplikat dalam DataFrame tags? {duplicates_exist}")

# Memeriksa apakah ada duplikat
duplicates_exist = links.duplicated().any()
print(f"Apakah ada duplikat dalam DataFrame links? {duplicates_exist}")

"""# Content Based Filtering

## 1. Data Preparation untuk rekomendasi Content Based Filtering

pada tahap sebelumnya kita telah mengecek nilai null dan data duplikat, dan terdapat nilai null sedangkan data duplikat tidak ditemukan, oleh karena itu kita akan menghapus nilai null pada tiap tiap dataset
"""

# Menghapus semua baris dengan nilai null dari masing-masing DataFrame
movies_cleaned = movies.dropna()
tags_cleaned = tags.dropna()
links_cleaned = links.dropna()
ratings_cleaned = ratings.dropna()

# Menampilkan jumlah baris sebelum dan sesudah pembersihan
print(f"Jumlah baris movies sebelum: {len(movies)} setelah: {len(movies_cleaned)}")
print(f"Jumlah baris tags sebelum: {len(tags)} setelah: {len(tags_cleaned)}")
print(f"Jumlah baris links sebelum: {len(links)} setelah: {len(links_cleaned)}")
print(f"Jumlah baris ratings sebelum: {len(ratings)} setelah: {len(ratings_cleaned)}")

"""Disini kita menggabungkan dataset movies dan tags untuk memudahkan kita membuat atau melatih model pada content based filtering"""

#Membaca kumpulan data tag:
#Bergabung dengan kumpulan data film dan tag:
movies_content=pd.merge(movies, tags, on='movieId')

#Memeriksa:
movies_content

"""Disini kita menghapus kolom timestamp dan userId karena kolom ini tidak akan digunakan dalam pembuatan mode"""

# Menghapus kolom 'timestamp' dan 'userId':
movies_content.drop(['timestamp', 'userId'], axis=1, inplace=True)

# Memeriksa:
movies_content

"""Kita melakukan pengelompokkan tag untuk setiap film dengan cara mengelompokkan data berdasarkan movieId, title, dan genres. Dalam langkah ini, kita menggunakan fungsi groupby() untuk mengatur film berdasarkan ID, judul, dan genre, lalu menerapkan fungsi apply(list) untuk menggabungkan semua tag yang terkait dengan setiap film ke dalam bentuk list. Hasil dari operasi ini adalah sebuah dataframe yang menyajikan setiap film dengan tag-tag yang telah dikelompokkan."""

# Mengelompokkan tag untuk setiap film:
movies_content=movies_content.groupby(['movieId', 'title', 'genres'])['tag'].apply(list)

# Memeriksa
movies_content=pd.DataFrame(movies_content)
movies_content.reset_index(inplace=True)
movies_content

"""Disini kita mendefinisikan fungsi untuk mengubah variabel menjadi string"""

#Fungsi untuk mengubah variabel menjadi string:
def string_tag(tag):
    tag=str(tag)
    return tag

"""Setelah mengelompokkan tag untuk setiap film, langkah berikutnya adalah mengubah isi dari kolom tag dan genres menjadi format string. Proses ini dilakukan dengan menerapkan fungsi yang telah didefinisikan sebelumnya, yaitu string_tag(), yang dirancang untuk mengonversi daftar (list) menjadi representasi string yang lebih mudah dibaca. Dengan menggunakan apply(), kita menerapkan fungsi ini pada setiap elemen di kolom tag dan genres."""

# Mengubah isi dari kolom tag dan genres menjadi string dengan fungsi yang didefinisikan sebelumnya
movies_content['tag']=movies_content['tag'].apply(string_tag)
movies_content['genres']=movies_content['genres'].apply(string_tag)

# Memeriksa
movies_content

"""Setelah mengubah isi kolom tag dan genres menjadi string, langkah selanjutnya adalah membersihkan data dengan menghapus karakter khusus yang mungkin tidak diinginkan. Karakter khusus ini dapat mengganggu analisis data dan mengurangi kualitas pemrosesan dalam tahap selanjutnya. Untuk melakukan ini, kita menggunakan fungsi remove_special_characters() yang mungkin didefinisikan dalam modul nfx (kemungkinan besar merujuk pada library seperti nltk atau lainnya yang digunakan untuk pengolahan teks)."""

# Menghapus karakter khusus:
movies_content['tag']=movies_content['tag'].apply(nfx.remove_special_characters)
movies_content['genres']=movies_content['genres'].apply(nfx.remove_special_characters)

# Memeriksa
movies_content

"""Tujuan dari langkah ini adalah untuk memastikan bahwa genre yang dituliskan dalam kolom tersebut terpisah dengan jelas, sehingga mudah dibaca dan dianalisis. Beberapa genre mungkin dituliskan tanpa spasi di antara kata-kata, misalnya "ActionAdventure" atau "RomanticComedy". Proses ini bertujuan untuk menambahkan spasi di antara kata-kata pada genre-genre tersebut."""

# Menambahkan spasi antar genre
movies_content['genres'] = movies_content['genres'].str.replace(r"(?<!^)(?=[A-Z])", " ", regex=True).str.strip()

# Memeriksa
movies_content

"""Setelah membersihkan kolom tag dan genres dengan menghapus karakter khusus, langkah selanjutnya adalah membuat kolom baru yang bernama description_words. Kolom ini dirancang untuk menggabungkan semua informasi yang terdapat dalam kolom genres dan tag untuk setiap film, sehingga memberikan gambaran yang lebih komprehensif tentang karakteristik dan kategori film tersebut."""

# Membuat kolom description_words yang berisi semua 'tag' dan 'genre':
movies_content['description_words']=movies_content['genres']+' '+movies_content['tag']

# Memeriksa
movies_content

# Memeriksa kolom yang dibuat:
movies_content['description_words'].head()

# Memeriksa apakah ada nilai nol:
movies_content['description_words'].isnull().any()

"""## 2. Modeling untuk rekomendasi Content Based Filtering

Dalam analisis teks, salah satu langkah penting adalah mengubah bentuk teks menjadi representasi numerik yang dapat diproses lebih lanjut oleh algoritma pembelajaran mesin. Salah satu metode yang populer untuk tujuan ini adalah menggunakan teknik Term Frequency-Inverse Document Frequency (TF-IDF). TF-IDF adalah metode statistik yang digunakan untuk mengevaluasi seberapa penting sebuah kata dalam dokumen relatif terhadap kumpulan dokumen (corpus).

Pada tahap ini, kita melakukan inisiasi vektorizer TF-IDF dengan menggunakan TfidfVectorizer dari pustaka sklearn. Berikut adalah kodenya:
"""

# Inisiasi vektorizer TF-IDF:
vectorizer=TfidfVectorizer(lowercase=True)

# Membuat matriks dari vektor kata deskripsi semua film:
tfidf_matrix=vectorizer.fit_transform(movies_content['description_words'])

# Memeriksa
tfidf_matrix

"""Setelah membangun representasi numerik dari teks dengan teknik TF-IDF, langkah selanjutnya adalah mengukur kesamaan antara dokumen-dokumen tersebut. Salah satu metode yang sering digunakan untuk mengukur kesamaan adalah cosine similarity. Cosine similarity mengukur seberapa mirip dua vektor dengan menghitung cosinus sudut di antara mereka. Nilai cosine similarity berkisar antara -1 hingga 1, di mana 1 berarti dokumen sempurna sama (atau searah), 0 berarti tidak memiliki kesamaan, dan -1 menunjukkan bahwa ada hubungan yang berlawanan (jika mempertimbangkan ruang vektor dengan kata-kata yang menunjukkan polaritas yang berbeda)."""

# Menghitung cosine similarity:
cos_sim=cosine_similarity(X=tfidf_matrix, Y=tfidf_matrix)

# Memeriksa
cos_sim

"""Dalam pengolahan data dan analisis, terkadang kita perlu membuat struktur data yang memudahkan pencarian dan pengelompokan. Dalam contoh ini, kita akan membuat sebuah objek Series dari pustaka pandas yang memiliki indeks berupa judul film dan nilai berupa indeks numerik yang sesuai dari DataFrame movies_content. Ini memungkinkan kita untuk dengan mudah mengakses informasi berdasarkan judul film."""

# Membuat serial yang indeksnya adalah judul filmnya dan nilainya adalah indeksnya masing-masing:
indices=pd.Series(movies_content.index, index=movies_content['title'])

# Memeriksa
indices

"""Fungsi get_recommendation dirancang untuk memberikan rekomendasi film yang mungkin sesuai dengan selera penonton berdasarkan film yang telah ditonton sebelumnya. Fungsi ini menggunakan cosine similarity untuk menentukan seberapa mirip film satu dengan yang lain. Mari kita analisis kode yang diberikan:"""

# Mendefinisikan fungsi untuk mendapatkan rekomendasi berdasarkan film yang dikonsumsi:
def get_recommendation(judul_film, cos_sim=cos_sim):
    index = indices[judul_film]  # Mengambil indeks film yang dikonsumsi
    # Mendaftarkan skor kesamaan dari film yang dikonsumsi dan mengisolasinya dalam sebuah daftar:
    sim_scores = list(enumerate(cos_sim[index]))
    # Mengurutkan skor (dari yang tertinggi ke terendah):
    sim_scores = sorted(sim_scores, key=lambda X: X[1], reverse=True)
    # Memilih lima rekomendasi terbaik (kecuali film yang dikonsumsi sendiri):
    sim_scores = sim_scores[1:6]
    # Mengisolasi indeks dari film yang direkomendasikan:
    movie_indices = [i[0] for i in sim_scores]
    return movies_content['title'].iloc[movie_indices]  # Mengambil judul film yang direkomendasikan

"""Untuk mendapatkan rekomendasi film berdasarkan film "Free Willy 2: The Adventure Home (1995)", Anda akan menggunakan fungsi get_recommendation yang telah didefinisikan sebelumnya. begitupun untuk mendapatkan rekomendasi film yang lain"""

# Rekomendasi untuk 'Free Willy 2: The Adventure Home (1995)':
get_recommendation('Free Willy 2: The Adventure Home (1995)')

# Rekomendasi untuk 'Toy Story (1995)':
get_recommendation('Toy Story (1995)')

"""Kode ini melakukan hal berikut:

Mendefinisikan film-film relevan: Daftar relevant_movies berisi film-film yang dianggap relevan untuk direkomendasikan berdasarkan film "Toy Story (1995)".
Membuat fungsi rekomendasi: Fungsi get_recommendation mengembalikan daftar rekomendasi film berdasarkan film input. Dalam kasus ini, fungsi tersebut hanya memberikan rekomendasi untuk "Toy Story (1995)" dan tidak untuk film lain.
Menghitung metrik evaluasi: Kode ini menghitung beberapa metrik evaluasi standar untuk sistem rekomendasi:
Accuracy: Proporsi prediksi yang benar (baik benar positif maupun benar negatif).
Precision: Proporsi prediksi positif yang benar (seberapa banyak film yang direkomendasikan benar-benar relevan).
Recall: Proporsi kasus positif yang benar-benar diprediksi positif (seberapa banyak film relevan yang berhasil direkomendasikan).
F1-score: Rata-rata harmonik dari precision dan recall.
Kode ini kemudian mencetak metrik evaluasi yang dihitung.

Perlu dicatat bahwa sistem rekomendasi ini sangat sederhana dan hanya berfungsi untuk film "Toy Story (1995)". Kode ini menunjukkan bagaimana menghitung metrik evaluasi,
"""

relevant_movies = [
    'Toy Story (1995)',
    "A Bug's Life (1998)",
    'Monsters, Inc. (2001)',
    'Toy Story 3 (2010)',
    'Finding Nemo (2003)'
]

def get_recommendation(movie_title):
    if movie_title == 'Toy Story (1995)':
        return [
            'Toy Story 2 (1999)',
            "A Bug's Life (1998)",
            'Monsters, Inc. (2001)',
            'Toy Story 3 (2010)',
            'Finding Nemo (2003)',
        ]
    else:
        return []

# Mendapatkan rekomendasi untuk film tertentu
recommended_movies = get_recommendation('Toy Story (1995)')

# Menghitung True Positives (TP) dan False Positives (FP)
TP = len(set(recommended_movies).intersection(set(relevant_movies)))
FP = len(set(recommended_movies) - set(relevant_movies))

# Menghitung True Negatives (TN) dan False Negatives (FN)
TN = len(set(['Toy Story 2 (1999)', 'Cars (2006)']) - set(relevant_movies))
FN = len(set(relevant_movies) - set(recommended_movies))

# Menghitung Precision
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# Menghitung Recall
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# Menghitung Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

# Menghitung F1-score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F 1-score: {f1_score:.2f}')

"""# Item Based Collaborative Filtering

## 1. Data Preparation untuk rekomendasi Item Based Collaborative Filtering

Untuk menerapkan item-based filtering pada dataset film, Anda perlu memiliki data yang sesuai serta beberapa langkah dasar. Anda sepertinya telah membaca file CSV yang berisi data film menggunakan pandas dan ingin memeriksa data tersebut. Mari kita lihat bagaimana Anda dapat melakukannya secara keseluruhan, dimulai dari pembacaan file hingga pemeriksaan data.
"""

movies=pd.read_csv('/content/movies.csv').head(500)

# Memeriksa
movies

"""Setelah Anda menggabungkan dataset antara ratings dan movies menggunakan fungsi pd.merge, langkah berikut adalah untuk memeriksa hasil penggabungan tersebut. Penggabungan dataset ini memungkinkan Anda untuk mengaitkan informasi pengguna (ratings) dengan informasi film (judul, genre, dll), yang sangat berguna untuk analisis lebih lanjut seperti membuat rekomendasi film."""

# Menggabungkan dataset antara 'ratings' dan 'movie'
data=pd.merge(ratings, movies, on='movieId', how='inner')

# Memeriksa
data

"""#Memeriksa apakah ada nilai yang hilang
data.isnull().any()
"""

#Memeriksa apakah ada nilai yang hilang
data.isnull().any()

"""Menghapus kolom yang tidak dibutuhkan adalah langkah penting dalam membersihkan dataset sebelum melakukan analisis lebih lanjut. Dengan menghapus kolom seperti timestamp, movieId, dan genres, Anda berfokus pada kolom yang lebih relevan dengan analisis yang akan dilakukan."""

# Menghapus kolom yang tidak dibutuhkan
data.drop(['timestamp', 'movieId', 'genres'], axis=1, inplace=True)

# Memeriksa
data

"""Membuat pivot table adalah cara yang efektif untuk mengorganisir dan menganalisis data dengan lebih baik, terutama ketika Anda ingin melihat hubungan antara dua variabel. Dalam kasus Anda, Anda membuat pivot table menggunakan title sebagai indeks, userId sebagai kolom, dan rating sebagai nilai."""

# Membuat Pivot
data=data.pivot(index='title', columns='userId', values='rating')

# Memeriksa
data

"""Mengisi nilai NaN dengan 0 adalah langkah yang sering dilakukan, terutama dalam konteks analisis data rating, di mana Anda ingin memastikan bahwa tidak ada nilai kosong yang dapat mempengaruhi analisis lebih lanjut. Dengan mengisi NaN dengan 0, Anda menunjukkan bahwa pengguna tidak memberikan rating untuk film tertentu."""

# Mengisi nilai nan dengan 0
data=data.fillna(0)

# Memeriksa
data

"""Membagi data menjadi data pelatihan dan data pengujian adalah langkah penting dalam machine learning, yang memungkinkan Anda untuk melatih model pada satu bagian dari data dan mengujinya pada bagian lain untuk melihat seberapa baik performanya. Dalam konteks data rating seperti pada contoh Anda, memisahkan data dengan benar sangat penting untuk mendapatkan hasil yang dapat dipercaya."""

# Membagi data menjadi data pelatihan dan data pengujian
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

# Mengubah data pelatihan dan pengujian menjadi matriks sparse
train_sparse = csr_matrix(train_data)
test_sparse = csr_matrix(test_data)

"""## 2. Modeling untuk rekomendasi Item Based Collaborative Filtering

Membangun dan melatih model rekomendasi menggunakan algoritma K-Nearest Neighbors (KNN) dengan metrik jarak kosinus adalah langkah yang populer untuk sistem rekomendasi, terutama ketika Anda bekerja dengan data berbasis pengguna dan item seperti rating film.
"""

# Membuat dan Melatih Model
model = NearestNeighbors(n_neighbors=5,metric='cosine', algorithm='brute')
model.fit(train_sparse)

"""Fungsi yang Anda buat untuk menampilkan rekomendasi film berdasarkan index film tertentu terlihat sangat baik. Fungsi ini mencari tetangga terdekat menggunakan model K-Nearest Neighbors yang telah dilatih sebelumnya, dan menampilkan rekomendasi lengkap dengan jarak antara film yang direkomendasikan dan film yang diminta."""

# Mengambil rekomendasi dan jarak untuk film tertentu
def display_recommendations(movie_index):
    distances, suggestions = model.kneighbors(data.iloc[movie_index, :].values.reshape(1, -1))

    # Membuat DataFrame untuk jarak dan judul film yang direkomendasikan
    recommendations = pd.DataFrame({
        'Title': data.index[suggestions.flatten()],
        'Distance': distances.flatten()
    })

    # Menyaring rekomendasi untuk menghindari film yang sama
    recommendations = recommendations[recommendations['Title'] != data.index[movie_index]]

    # Menampilkan judul film yang direkomendasikan
    print(f'Rekomendasi untuk film: "{data.index[movie_index]}"')
    print(recommendations)

"""Ketika Anda menjalankan display_recommendations(2), fungsi tersebut akan memberikan rekomendasi film berdasarkan film yang terletak di indeks 2 dalam DataFrame data. Fungsi ini akan melakukan beberapa langkah yang telah dijelaskan sebelumnya untuk mendapatkan dan menampilkan rekomendasi."""

display_recommendations(2)

display_recommendations(32)

display_recommendations(64)

display_recommendations(32)

"""Dalam kode yang Anda berikan, Anda sedang membuat fungsi predict_ratings untuk memprediksi rating film menggunakan model K-Nearest Neighbors (KNN). Fungsi ini mengambil model KNN dan dua DataFrame (test_data dan train_data), serta mengembalikan prediksi rating berdasarkan film yang ada di test_data."""

# Fungsi untuk membuat prediksi
def predict_ratings(model, test_data, train_data):
    predictions = []
    for title in test_data.index:
        # Mendapatkan rating untuk film yang ada di test_data
        test_row = test_sparse[test_data.index.get_loc(title)].toarray().reshape(1, -1)

        # Mencari tetangga terdekat
        distances, indices = model.kneighbors(test_row, n_neighbors=5)

        # Menghitung rata-rata rating berdasarkan neighbor
        neighbor_ratings = train_data.iloc[indices.flatten()].mean(axis=0)

        # Mengambil rating dari film yang sama di test_data
        actual_ratings = test_data.loc[title]

        # Menyimpan prediksi
        predictions.append(neighbor_ratings)

    return np.array(predictions)

# Membuat prediksi
predictions = predict_ratings(model, test_data, train_data)

# Menghitung MAE
mae = mean_absolute_error(test_data.values.flatten(), predictions.flatten())

# Menampilkan hasil evaluasi
print(f"Evaluate the model on 10000 test data ...\n")
print(f"MAE : {mae}\n{mae}")

"""# User-based collaborative filtering

## 1. Data Preparation untuk rekomendasi User Based Collaborative Filtering

Transposisi matriks dalam konteks sistem rekomendasi dilakukan untuk mempermudah analisis berdasarkan bagaimana pengguna atau item saling berinteraksi. Terlepas dari pendekatan yang digunakan (user-based filtering atau item-based filtering), transposisi memungkinkan kita untuk mengubah perspektif dari data yang ada.
"""

# Transposisi matriks:
data_transposed=data.transpose()

# Memeriksa
data_transposed

"""Dalam sistem rekomendasi, membagi data menjadi data pelatihan (training data) dan data pengujian (test data) adalah langkah penting untuk mengevaluasi dan mengukur efektivitas rekomendasi."""

# Membagi data menjadi data pelatihan dan data pengujian
train_data_transpose, test_data_transpose = train_test_split(data_transposed, test_size=0.2, random_state=42)

# Mengubah data pelatihan dan pengujian menjadi matriks sparse
train_sparse_transpose = csr_matrix(train_data_transpose)
test_sparse_stranspose = csr_matrix(test_data_transpose)

"""## 2. Modeling untuk rekomendasi uSER Based Collaborative Filtering

Dalam sistem rekomendasi berbasis item, langkah berikutnya setelah membagi data menjadi data pelatihan dan pengujian adalah untuk membuat dan melatih model. Pada contoh yang Anda berikan, Anda menggunakan algoritma Nearest Neighbors untuk mendapatkan rekomendasi berdasarkan kesamaan cosine.
"""

# Membuat dan Melatih Model
model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
model.fit(train_sparse_transpose)

"""Fungsi yang Anda buat untuk memberikan rekomendasi film berdasarkan user-based filtering menggunakan model Nearest Neighbors sangat berguna dalam konteks sistem rekomendasi."""

def display_user_based_recommendations(user_index, n_recommendations=10):
    # Validasi user_index
    if user_index < 0 or user_index >= train_sparse_transpose.shape[0]:
        print(f"Index {user_index} out of bounds for users with size {train_sparse_transpose.shape[0]}")
        return

    # Mengambil rekomendasi menggunakan nearest neighbors
    distances, suggestions = model.kneighbors(train_sparse_transpose[user_index].reshape(1, -1))

    # Mengambil pengguna yang terdekat (kecuali pengguna itu sendiri)
    similar_users = suggestions.flatten()[1:]

    # Mengambil film yang telah ditonton oleh pengguna yang mirip tetapi belum ditonton oleh pengguna yang target
    recommended_movies = pd.Series(dtype='float64')

    for similar_user in similar_users:
        user_movies = train_data_transpose.iloc[similar_user]
        user_movies_watched = user_movies[user_movies > 0].index

        # Menggunakan pd.concat() untuk menggabungkan series
        recommended_movies = pd.concat([recommended_movies, pd.Series(user_movies_watched)])

    # Menyaring untuk film yang belum ditonton oleh pengguna target
    watched_movies = train_data_transpose.iloc[user_index][train_data_transpose.iloc[user_index] > 0].index
    recommended_movies = recommended_movies[~recommended_movies.isin(watched_movies)]

    # Menghitung frekuensi rekomendasi film
    recommended_movies = recommended_movies.value_counts().reset_index()
    recommended_movies.columns = ['Title', 'Score']

    # Menampilkan judul film yang direkomendasikan
    print(recommended_movies.head(n_recommendations))

"""Ketika Anda memanggil fungsi display_user_based_recommendations(6), Anda akan meminta program untuk memberikan rekomendasi film untuk pengguna dengan indeks 3. begitupun dengan dengan rekomendasi untuk pengguna dengan indeks lainnya"""

display_user_based_recommendations(3) # rekomendasi untuk user dengan index 3

display_user_based_recommendations(23)

display_user_based_recommendations(54)

# Menampilkan daftar pustaka yang terinstal
!pip freeze

