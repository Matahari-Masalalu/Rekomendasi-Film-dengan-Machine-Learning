# Laporan Proyek Machine Learning - Faishal Anwar Haysim

## Rekomendasi Film dengan Machine Learning

## Domain Proyek
Proyek ini bertujuan untuk membangun sistem rekomendasi film menggunakan teknik machine learning. Sistem ini akan menggunakan dataset film yang mencakup informasi mengenai films, tags, ratings, dan links. Dengan banyaknya pilihan film yang tersedia, sistem rekomendasi diharapkan dapat membantu pengguna menemukan film yang sesuai dengan preferensi mereka, meningkatkan pengalaman menonton, dan mengurangi waktu yang dihabiskan untuk mencari film.

## Business Understanding

### Problem Statements

- Pernyataan Masalah: Bagaimana merancang sistem rekomendasi film yang dapat memberikan rekomendasi akurat berdasarkan preferensi pengguna?

### Goals
- Jawaban pernyataan masalah: Membangun sistem rekomendasi film menggunakan teknik Content-Based Filtering dan Collaborative Filtering untuk memberikan rekomendasi yang relevan kepada pengguna.
### Solution Statement
- Penggunaan Content-Based Filtering: Menggunakan informasi dari deskripsi film, genre, dan tag untuk memberikan rekomendasi film yang mirip.
- Penggunaan Collaborative Filtering: Menggunakan data rating pengguna untuk memberikan rekomendasi berdasarkan perilaku pengguna lain yang memiliki preferensi serupa.

## Data Understanding

### Sumber Data
sumber/referensi :  https://www.kaggle.com/datasets/kanametov/movies-recomendation-system

### Jumlah Data
Dataset yang digunakan dalam proyek ini terdiri dari beberapa file CSV, yaitu:
- movies.csv: 34,208 baris dan 3 kolom (movieId, title, genres).
- links.csv: 34,208 baris dan 2 kolom (movieId, imdbId).
- tags.csv: 34,208 baris dan 4 kolom (userId, movieId, tag, timestamp).
- ratings.csv: 22,884,377  baris dan 4 kolom (userId, movieId, rating, timestamp).

### Penjelasan Fitur Dataset
- movieId: ID unik untuk setiap film.
- title: Judul film
- genres: Genre film, yang dapat terdiri dari beberapa genre yang dipisahkan oleh tanda '|'.
- imdbId: ID film di IMDb untuk referensi lebih lanjut.
- userId: ID unik untuk setiap pengguna yang memberikan rating.
- tag: Tag atau label yang diberikan oleh pengguna untuk film tertentu.
- rating: Rating yang diberikan oleh pengguna untuk film, biasanya dari 1 hingga 5.
- timestamp: Waktu ketika rating atau tag diberikan.

### Kondisi Data
- Missing Values: Setelah dilakukan analisis ada nilai yang hilang(null) pada dataset tags, links, dan rating.
- Duplikat: Setelah dilakukan analisis tidak ditemukan duplikat dalam tiap-tiap dataset.

### Exploratory Data Analysis
Pada tahap ini, kita melakukan eksplorasi data untuk mendapatkan wawasan awal mengenai dataset. Beberapa analisis yang dilakukan meliputi:
- Genre Film yang Paling Umum: Mengidentifikasi genre film yang paling banyak ditonton.
![Cuplikan layar 2024-11-09 101428](https://github.com/user-attachments/assets/994903d6-ab6f-4487-a447-cbbc48b33330)
Genre film yang paling umum adalah Drama dan Comedy, yang menunjukkan preferensi pengguna terhadap film-film tersebut.


- Tag yang Sering Digunakan: Menganalisis tag yang sering digunakan oleh pengguna.
![Cuplikan layar 2024-11-09 101547](https://github.com/user-attachments/assets/b97c394c-ab9f-4f0c-a148-d375b9175f4b)
Tag yang sering digunakan memberikan wawasan tentang tema atau elemen yang disukai pengguna dalam film.


- Menganalisis tag yang sering digunakan oleh pengguna.
![Cuplikan layar 2024-11-09 101631](https://github.com/user-attachments/assets/9ea03ecd-0b18-42fd-b0d2-156f71610390)

Dari distribusi rating, terlihat bahwa mayoritas film mendapatkan rating di atas 3, menunjukkan kepuasan pengguna yang relatif tinggi.

## Data Preparation
### 1. Data Preparation Pada Content-Based Filtering:
- Menghapus Nilai Null: menghapus baris dengan nilai null dari dataset untuk memastikan hanya data lengkap yang digunakan dalam analisis. Baris tanpa tag dalam dataset tags dihapus, sedangkan baris tanpa rating dalam dataset ratings juga dihilangkan. Langkah ini menjamin relevansi dan kualitas data yang digunakan.
- Penggabungan Dataset: Menggabungkan dataset ratings.csv dan tags.csv berdasarkan movieId.
- Penghapusan Kolom : Menghapus kolom timestamp dan userId pada data yang telah digabungkan sebelumnya.
- Mengelompokkan Tag untuk Setiap Film: Mengelompokkan tag untuk setiap film berdasarkan movieId, title, dan genres. Ini menyusun tag menjadi list untuk setiap film, sehingga kita dapat melihat semua tag yang terkait dengan film tersebut.
- Menghapus Karakter Khusus: Pada langkah ini, kita akan menggunakan fungsi remove_special_characters untuk menghapus karakter khusus dari kolom tag dan genres. Ini membantu memastikan bahwa data yang kita gunakan tidak mengandung karakter yang tidak diinginkan yang dapat mempengaruhi analisis atau pemodelan.
- Menambahkan Spasi Antar Genre: Dalam beberapa dataset, genre mungkin ditulis tanpa spasi antar kata (misalnya, "ActionAdventure"). Untuk meningkatkan keterbacaan dan konsistensi, kita menambahkan spasi antar genre. Kita menggunakan regex untuk menambahkan spasi sebelum huruf kapital yang tidak berada di awal string.
- ** Membuat kolom description_words yang berisi semua 'tag' dan 'genre' untuk setiap film, yang akan digunakan dalam proses modeling.
- TF-IDF Vectorization: Menggunakan TfidfVectorizer untuk mengubah kolom description_words menjadi representasi numerik. Parameter lowercase=True digunakan untuk mengubah semua teks menjadi huruf kecil.
### 2. Data Preparation Pada Collaborative Filtering:
- Penggabungan Dataset: Menggabungkan dataset ratings.csv dan movies.csv berdasarkan movieId.
- Pembuatan Pivot Table: Membuat pivot table yang menunjukkan rating yang diberikan oleh pengguna untuk setiap film.
- Mengisi Nilai yang Hilang: Mengisi nilai yang hilang (NaN) dengan 0, yang menunjukkan bahwa pengguna tersebut tidak memberikan rating untuk film tertentu.
- Transposisi Matriks: Untuk keperluan analisis lebih lanjut, matriks transposisi dibuat sehingga baris mewakili pengguna dan kolom mewakili film. Ini mempermudah saat melakukan rekomendasi berbasis pengguna.
- Membagi Data: Data dibagi menjadi data pelatihan (training) dan data pengujian (test) menggunakan train_test_split. Ini bertujuan untuk melatih model pada satu bagian data dan mengujinya pada bagian lain untuk mengevaluasi performa model.
- Ekstraksi Fitur Menggunakan Matriks Sparse: Untuk efisiensi memori dan kecepatan komputasi, matriks sparse digunakan. Matriks sparse menyimpan hanya nilai-nilai yang tidak nol, yang sangat berguna dalam konteks sistem rekomendasi di mana sebagian besar pengguna tidak memberikan rating untuk sebagian besar film.

## Modeling
Pada proyek ini, dua pendekatan digunakan untuk memberikan rekomendasi film, yaitu Content-Based Filtering dan Collaborative Filtering. Kedua pendekatan ini memanfaatkan teknik machine learning untuk menganalisis data dan menghasilkan rekomendasi yang relevan bagi pengguna.
### 1. Content-Based Filtering:
Content-Based Filtering adalah pendekatan dalam sistem rekomendasi yang berfokus pada analisis konten dari item (dalam hal ini, film) untuk memberikan rekomendasi kepada pengguna. Pendekatan ini menggunakan informasi yang tersedia tentang item, seperti deskripsi, genre, dan tag, untuk menentukan kesamaan antara item yang berbeda. Dengan demikian, rekomendasi yang dihasilkan akan relevan dengan preferensi pengguna yang telah ditentukan oleh film yang mereka tonton sebelumnya.

1. Cosine Similarity: Langkah pertama dalam Content-Based Filtering adalah menghitung cosine similarity untuk menemukan kesamaan antar film. Cosine similarity mengukur sudut antara dua vektor dalam ruang multidimensi, yang dalam konteks ini adalah representasi numerik dari film berdasarkan fitur yang diekstrak dari genre atau deskripsi film. Dengan menghitung cosine similarity, kita dapat menentukan seberapa mirip dua film berdasarkan konten yang mereka miliki. Nilai cosine similarity berkisar antara 0 (tidak ada kesamaan) hingga 1 (kesamaan sempurna), sehingga memungkinkan kita untuk mengidentifikasi film-film yang memiliki deskripsi atau genre yang serupa.
   
2. Fungsi Rekomendasi: Setelah menghitung cosine similarity, langkah berikutnya adalah membuat fungsi get_recommendation. Fungsi ini akan memberikan rekomendasi film berdasarkan film yang telah ditonton oleh pengguna. Dengan menggunakan hasil cosine similarity, fungsi ini dapat mencari film yang paling mirip dengan film yang dipilih oleh pengguna. Sebagai contoh, jika pengguna telah menonton dan menyukai film tertentu, fungsi ini akan mengembalikan daftar film lain yang memiliki kesamaan konten, sehingga meningkatkan kemungkinan bahwa pengguna akan menikmati film-film tersebut.
  
### 2. Collaborative Filtering:
Collaborative Filtering adalah pendekatan dalam sistem rekomendasi yang memanfaatkan data interaksi pengguna untuk menemukan film yang mirip berdasarkan rating yang diberikan oleh pengguna. Pendekatan ini berfokus pada pola perilaku pengguna dan mengidentifikasi film yang mungkin disukai pengguna berdasarkan preferensi pengguna lain yang memiliki kesamaan.

1. Dalam pendekatan Collaborative Filtering, langkah pertama adalah menghitung kesamaan antara film berdasarkan rating yang diberikan oleh pengguna. Metrik yang umum digunakan untuk mengukur kesamaan ini adalah cosine similarity atau Pearson correlation. Pada kasus ini, saya menggunakan cosine similarity untuk mengukur seberapa mirip rating yang diberikan oleh pengguna terhadap film-film yang berbeda. Dengan menghitung kesamaan ini, kita dapat menentukan hubungan antara film berdasarkan preferensi pengguna, yang kemudian akan digunakan untuk memberikan rekomendasi.

2. Mencari Item Mirip: Setelah menghitung kesamaan antara film, langkah selanjutnya adalah mencari film lain yang memiliki rating serupa. Film-film ini disebut sebagai "tetangga terdekat" untuk film tertentu. Proses ini melibatkan pengidentifikasian film yang memiliki kesamaan rating dengan film yang sedang dianalisis, sehingga memungkinkan sistem untuk merekomendasikan film yang mungkin disukai pengguna berdasarkan film yang telah mereka tonton dan beri rating sebelumnya.

3. Rekomendasi diberikan berdasarkan film yang mirip dengan film yang telah ditonton oleh pengguna. Jika pengguna telah menonton dan menyukai film tertentu, maka film-film yang mirip akan direkomendasikan. Fungsi ini akan mengembalikan daftar film yang dapat direkomendasikan kepada pengguna berdasarkan kesamaan rating yang telah dihitung sebelumnya.


Parameter yang digunakan dalam Model Collaborative Filtering: adalah sebagai berikut::
- n_neighbors=5: Menentukan jumlah tetangga terdekat (film) yang akan dicari.
- metric='cosine': Menggunakan cosine similarity sebagai metode pengukuran kesamaan.
- algorithm='brute': Menentukan algoritma brute-force untuk menemukan tetangga terdekat, cocok untuk dataset kecil atau sedang.
  

### Hasil Prediksi

#### Rekomendasi Pada Content Based Filtering
  
| movieId | title                                         |
|---------|-----------------------------------------------|
|   835   |                        Shaggy Dog, The (1959) |
|   398   |                             Free Willy (1993) |
|   4672  |                 Look Who's Talking Now (1993) |
|   834   | Homeward Bound: The Incredible Journey (1993) |
|  21089  |                                   Noi 4 (2014 |

Rekomendasi di atas diambil berdasarkan kesamaan karakteristik dengan film "Free Willy 2: The Adventure Home (1995)". Dengan menggunakan metode Content-Based Filtering, model berusaha untuk mencocokkan film yang memiliki genre dan tema yang mirip, sehingga dapat memberikan pengalaman menyenangkan dan sesuai dengan preferensi.

#### Rekomendasi Pada Item Based Filtering

|   | title                                     | title    |
|---|-------------------------------------------|----------|
| 0 | Boxing Helena (1993)                      | 0.309421 |
| 1 | Miracle on 34th Street (1994)             | 0.378249 |
| 2 | Lawnmower Man 2: Beyond Cyberspace (1996) | 0.400886 |
| 3 | Carrington (1995)                         | 0.407813 |
| 4 | Steal Big, Steal Little (1995)            | 0.410505 |

Rekomendasi di atas diambil berdasarkan kesamaan karakteristik film dengan indeks 2 yaitu ""Ace Ventura: Pet Detective (1994)"". Dengan menggunakan metode Item-Based Filtering, model berusaha untuk mencocokkan film yang memiliki rating mirip, sehingga dapat memberikan pengalaman menyenangkan dan sesuai dengan preferensi.



#### Rekomendasi Pasa User Based Filtering

|   | title                                             | Score |
|---|---------------------------------------------------|-------|
| 0 | Apollo 13 (1995)                                  | 4     |
| 1 | Crimson Tide (1995)                               | 4     |
| 2 | GoldenEye (1995)                                  | 4     |
| 3 | Quiz Show (1994)                                  | 4     |
| 4 | Interview with the Vampire: The Vampire Chroni... | 3     |
| 5 | Ace Ventura: When Nature Calls (1995)             | 3     |
| 6 | Lion King, The (1994)                             | 3     |
| 7 | Babe (1995)                                       | 2     |
| 8 | While You Were Sleeping (1995)                    | 2     |
| 9 | Disclosure (1994)                                 | 1     |

Rekomendasi di atas diambil berdasarkan kesamaan karakteristik film dengan User dengan id (3). Dengan menggunakan metode User-Based Filtering, model berusaha untuk mencocokkan film yang memiliki rating mirip, sehingga dapat memberikan pengalaman menyenangkan dan sesuai dengan preferensi.

## Evaluation

### Evaluasi Pada Conten-Based Filtering
Untuk evaluasi pada Content-Based Filtering saya menggunakan beberapa metrik yaitu Accuracy, Precision, Recall, dan F1-Score
-Akurasi Akurasi mengukur proporsi prediksi yang benar dari semua prediksi yang dibuat oleh model. Dalam sistem rekomendasi, akurasi dapat diartikan sebagai persentase rekomendasi yang tepat yang diberikan kepada pengguna.
- Presisi Presisi mengukur proporsi rekomendasi yang benar di antara semua rekomendasi yang diberikan oleh model. Metrik ini penting untuk memastikan bahwa rekomendasi yang diberikan kepada pengguna benar-benar sesuai dengan apa yang mereka sukai.
- Recall Recall mengukur proporsi rekomendasi yang benar di antara semua rekomendasi yang sebenarnya relevan. Metrik ini penting untuk memastikan bahwa model tidak melewatkan rekomendasi yang penting.
- F1-score F1-score merupakan nilai harmonik dari presisi dan recall. Metrik ini memberikan gambaran yang lebih lengkap tentang performa model dibandingkan dengan hanya menggunakan presisi atau recall saja.

Accuracy, Precision, Recall, dan F1-Score dihitung dengan rumus berikut:


![image](https://github.com/user-attachments/assets/b90dd806-9141-4a0d-88a5-d93ffd13a8c3)


Di mana:

TP (True Positive): Jumlah rekomendasi yang benar dan relevan.

TN (True Negative): Jumlah rekomendasi yang benar dan tidak relevan.

FP (False Positive): Jumlah rekomendasi yang salah dan tidak relevan.

FN (False Negative): Jumlah rekomendasi yang salah dan relevan.

#### Hasil Evaluasi Menggunakan Accuracy, Precision, Recall, dan F1-Score

![Cuplikan layar 2024-11-10 111811](https://github.com/user-attachments/assets/33134969-1ac0-40fc-9fdd-b99612e01b31)

Secara keseluruhan, hasil evaluasi menunjukkan bahwa model memiliki performa yang baik dalam memberikan rekomendasi. Model cukup akurat dalam memprediksi rekomendasi yang relevan, dan berhasil menemukan sebagian besar rekomendasi yang relevan yang mungkin diberikan.

### Eavluasi Pada Collaborative Filtering
Evaluasi Model Rekomendasi Menggunakan Mean Absolute Error (MAE)

Mean Absolute Error (MAE) adalah salah satu metode evaluasi yang umum digunakan dalam sistem rekomendasi untuk mengukur kesalahan prediksi model. MAE memberikan informasi tentang seberapa dekat prediksi model dengan nilai aktual yang ada, dalam hal ini, rating yang diberikan oleh pengguna.

Definisi MAE
MAE dihitung sebagai rata-rata dari selisih absolut antara nilai yang diprediksi (prediksi rating) dan nilai yang sebenarnya (rating aktual). Rumus untuk menghitung MAE adalah sebagai berikut:


![Cuplikan layar 2024-11-10 013916](https://github.com/user-attachments/assets/a3c76aef-f44d-4028-a051-c1bb32948fe2)


Di mana:

( n ) = jumlah total prediksi (data pengujian).

( yi ) = rating aktual dari item (film) yang ditonton oleh pengguna (data tes).

( ŷ ) = rating yang diprediksi oleh model.

### Hasil Evaluasi Menggunakan MAE


![Cuplikan layar 2024-11-10 014151](https://github.com/user-attachments/assets/ea56af71-22e3-4f00-8961-bb94435a71bc)


Secara keseluruhan, MAE sebesar 0.20173630200422316 menunjukkan bahwa model rekomendasi film dapat memberikan prediksi rating yang cukup akurat, dan pengguna kemungkinan besar akan merasa puas dengan rekomendasi yang diberikan berdasarkan model tersebut. Namun, selalu penting untuk membandingkan MAE ini dengan nilai MAE dari model lain atau baseline untuk menilai seberapa baik model Anda dibandingkan dengan alternatif yang ada.

## Dampak Model terhadap Pemahaman Bisnis:

Hasil evaluasi sistem rekomendasi film menunjukkan bahwa model dapat menjawab pernyataan masalah yang diajukan di awal. Dengan menggunakan teknik Content-Based Filtering dan Collaborative Filtering, sistem ini berhasil memberikan rekomendasi film yang relevan berdasarkan preferensi pengguna dan perilaku rating pengguna lain.

Sistem rekomendasi ini memiliki dampak signifikan terhadap pengalaman pengguna dalam menemukan film yang sesuai dengan minat mereka. Dengan meningkatkan relevansi rekomendasi, pengguna dapat menghabiskan lebih sedikit waktu untuk mencari film dan lebih banyak waktu untuk menikmati konten yang mereka sukai.

Selain itu, informasi yang dihasilkan dari sistem ini sangat berharga bagi platform streaming dan penyedia konten. Dengan memahami preferensi pengguna melalui analisis data, mereka dapat mengoptimalkan katalog film mereka, meningkatkan kepuasan pengguna, dan pada akhirnya meningkatkan retensi pelanggan.

Secara keseluruhan, model rekomendasi ini tidak hanya memberikan nilai tambah bagi pengguna, tetapi juga berkontribusi pada strategi bisnis yang lebih luas dengan meningkatkan engagement dan loyalitas pengguna terhadap platform.

## Kesimpulan
Proyek ini berhasil membangun sistem rekomendasi film menggunakan teknik Content-Based Filtering dan Collaborative Filtering. Hasil evaluasi menunjukkan bahwa sistem dapat memberikan rekomendasi yang relevan dan membantu pengguna dalam menemukan film yang sesuai dengan preferensi mereka.

## Referensi
1. Dicoding. Diakses pada 10 November 2024 dari https://www.dicoding.com/academies/319-machine-learning-terapan
2. Rodrigo de Menezes Gomes - Kaggle. Diakses pada 10 November 2024 dari https://www.kaggle.com/datasets/kanametov/movies-recomendation-system
3. Rodrigo de Menezes Gomes - Kaggle. Diakses pada 10 November 2024 dari https://www.kaggle.com/code/rodmnzs/movies-recommendation-system
4. Google Colab. Diakses pada 10 November 2024 https://g.co/kgs/tPKUQUM
