# Laporan Proyek 1 Machine Learning - Muhammad Hidayat Mauluddin

## Domain Proyek

Industri Properti adalah salah satu industri yang sangat menjanjikan dari segi bisnis. Dikarenakan properti seperti tempat tinggal adalah salah satu kebutuhan primer bagi manusia[1]. Seiring bertambahnya populasi manusia, permintaan terhadap properti juga akan semakin banyak dan dapat mengakibatkan peningkatan harga. Dalam dunia industri properti yang sangat dinamis, memiliki kemampuan untuk memprediksi harga properti dapat memberikan keuntungan kompetitif yang signifikan. Terkadang seseorang salah dalam menentukan harga jual properti. Dan itu mengakibatkan kerugian seperti properti lama terjual dengan harga murah, kerugian dari segi materi dikarenakan harga jual yang terlalu rendah[3]. Hal itu terjadi dikarenakan kurangnya pengetahuan tentang fitur-fitur atau faktor-faktor yang mempengaruhi harga jual properti seperti lokasi, ukuran properti, kondisi, usia properti, fasilitas, jumlah kamar tidur, jumlah kamar, atau fasilitas lainnya yang dapat mempengaruhi besar kecilnya harga jual properti.

Berlandaskan dari permasalahan yang dijelaskan di atas, kita dapat menggunakan algoritma *Machine Learning* sebagai solusi untuk permasalahan tersebut. *Predictive Analysis* adalah model machine learning yang memanfaatkan data untuk dipelajari dan dapat menghasilkan prediksi. Dengan teknik *Predictive Analysis*, *Machine Learning* membantu perusahaan dengan cara mempelajari histori data penjualan dengan fitur-fitur yang ada. Dan dengan menggunakan teknik *predictive analysis* dan algoritma *Machine Learning*, diharapkan perusahaan penjual properti dapat memahami faktor-faktor yang mempengaruhi harga properti seperti luas area, jumlah kamar tidur, dan jumlah kamar mandi. Penjual dapat menentukan harga yang kompetitif dan realistis untuk produk mereka. Ini dapat membantu mereka dalam penetapan harga yang lebih efektif, strategi pemasaran yang tepat, serta memberikan perspektif yang lebih baik tentang permintaan dan tren pasar. Dengan menggunakan prediksi harga properti, penjual juga dapat mengoptimalkan laba dan mengelola inventaris mereka dengan lebih baik.



## Business Understanding

### Problem Statements

1. Bagaimana mengidentifikasi fitur-fitur yang signifikan yang mempengaruhi harga jual rumah atau properti?
2. Bagaimana mengembangkan model prediksi harga jual rumah atau properti berdasarkan fitur-fitur tersebut?

### Goals

1. Mengidentifikasi fitur-fitur yang memiliki pengaruh signifikan terhadap harga jual rumah atau properti.
2. Mengembangkan model prediksi harga jual rumah atau properti yang dapat memberikan estimasi harga yang akurat.

### Solution Statements

1. Melakukan analisis eksplorasi data untuk mengidentifikasi fitur-fitur yang berkorelasi kuat dengan harga jual rumah atau properti.
2. Menangani data yang hilang atau tidak lengkap melalui teknik pengisian atau penghapusan yang tepat.
3. Mengatasi adanya *outliers* dalam data menggunakan metode yang sesuai seperti IQR.
4. Menggunakan metode reduksi dimensi seperti PCA untuk mengurangi dimensi data jika diperlukan.
5. Menggunakan algoritma regresi seperti Linear Regression, K-Nearest Neighbors, Random Forest, atau Boosting Algorithm untuk mengembangkan model prediksi harga.
6. Melakukan evaluasi dan validasi model menggunakan metrik yang relevan seperti Mean Squared Error (MSE).
7. Menerapkan model prediksi harga pada data baru untuk mendapatkan estimasi harga jual rumah atau properti yang akurat.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah dataset harga jual rumah atau properti di Thailand, Bangkok. Dataset ini terdiri dari 562 baris (observasi) dan 5 kolom (variabel). Dataset ini sudah dalam kondisi yang cukup bersih sehingga tidak memerlukan banyak proses pembersihan data.

Berikut ini adalah informasi lebih terperinci tentang dataset:

- Jumlah baris (observasi): 562

- Jumlah kolom (variabel): 5

- Variabel yang terdapat dalam dataset:

  - Harga (Price): Merupakan variabel target yang menunjukkan harga jual rumah atau properti dalam mata uang tertentu.

  - Luas Bangunan (Area (sq. ft.)): Merupakan luas bangunan rumah atau properti dalam satuan tertentu.

  - Jumlah Kamar Tidur (Bedrooms): Merupakan jumlah kamar tidur yang dimiliki oleh rumah atau properti.

  - Jumlah Kamar Mandi (Bathrooms): Merupakan jumlah kamar mandi yang dimiliki oleh rumah atau properti.

  - Lokasi (Location): Merupakan lokasi rumah atau properti di Thailand, Bangkok.

### EDA - Handling Outliers:

Berdasarkan visualisasi boxplot, terdapat beberapa outliers pada fitur-fitur tersebut.

- *Numerical Feature*: Area (sq. ft.)

![Numerical Feature Area (sq  ft )](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/77384f34-8cf5-409c-9db3-9442251f49d7)

  Gambar 1. *Numerical Feature*: Area (sq. ft.)

- *Numerical Feature*: *Bedrooms*

  ![Numerical Feature Bedrooms](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/ad3edd44-a32d-412e-8006-62fa83cdea9d)

  Gambar 2. *Numerical* *Feature* *Bedrooms*

- *Numerical Feature: Bathrooms*

![Numerical Feature Bathrooms](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/2eec8f42-3965-46cc-b6d9-2773984daf01)

  Gambar 3. *Numerical Feature Bathrooms*

- *Numerical Feature*: *Price* (THB)

![Numerical Feature Price (THB)](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/2c3a519d-3c5f-4e79-94c1-4cd214cb4910)

  Gambar 4. Numerical Feature: Price (THB)

Untuk mengatasi outliers tersebut, akan dilakukan penghapusan outliers dengan menggunakan metode IQR (*Interquartile Range*).

### EDA - Univariate Analysis

- #### Categorical Feature:

1. *Categorical Feature Property Type*


   Tabel 1. *Categorical Feature Property Type*

   |          | Jumlah Sampel | Persentase |
   | -------- | ------------- | ---------- |
   | Condo    | 198           | 35.2       |
   | House    | 190           | 33.7       |
   | Apartmen | 175           | 31.1       |

![Categorical Feature - Property Type](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/6a2d90b9-8a19-4c6f-9a90-bbbb283081a2)

​	Gambar 5. Diagram *Categorical Feature Property Type*

Berdasarkan Tabel 1 dan Gambar 5 di atas, dapat dilihat informasi mengenai data *Property Type* beserta persentasenya.

2. *Categorical Feature Location*

   Tabel 2. *Categorical Feature Location*

   |             | jumlah sampel | persentase |
   | ----------- | ------------- | ---------- |
   | Ladprao     | 63            | 11.2       |
   | Siam        | 62            | 11.0       |
   | Sukhumvit   | 61            | 10.8       |
   | Silom       | 58            | 10.3       |
   | Phrom Phong | 55            | 9.8        |
   | Thonglon    | 55            | 9.8        |
   | Ari         | 55            | 9.8        |
   | Ekkamai     | 55            | 9.8        |
   | Sathorn     | 52            | 9.2        |
   | Ratchada    | 47            | 8.3        |


![Categorical Feature - Location](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/90bf6e44-dd2c-477e-93cc-0346d631c6c8)

​	Gambar 6. Diagram *Categorical Feature Location*

Berdasarkan Tabel 2 dan Gambar 6 di atas, dapat dilihat informasi mengenai data *Location* beserta persentasenya.



1. *Numerical Feature*:

![Histogram Numerical Feature](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/cae4a297-519e-402a-b88e-fce57308fd88)

   Gambar 7. *Histogram Numerical Feature*

Mari kita perhatikan Gambar 7, yaitu histogram *Numerical Feature*. Dari gambar tersebut, dapat diperoleh beberapa informasi berikut:

  - Terdapat peningkatan harga properti yang disertai dengan penurunan jumlah sampel pada kolom *Price*.
  - Distribusi harga properti cenderung miring ke kanan (*right-skewed*). Hal ini menunjukkan bahwa ada sebagian besar properti dengan harga yang relatif rendah, sedangkan properti dengan harga tinggi lebih sedikit. Informasi ini perlu dipertimbangkan saat membangun model prediksi, karena kemiringan distribusi dapat mempengaruhi performa model.

### EDA - Multivariate Analyis

#### Categorical Feature terhadap price: 

![Categorical Feature terhadap price](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/2e56323c-9ffb-43b6-929f-f837cc36b730)

Gambar 8. *Categorical Feature* terhadap *price*

Berdasarkan gambar tersebut, dapat disimpulkan:

1. Rata-rata tertinggi pada fitur *Property Type* adalah *Rumah*.
2. Rata-rata tertinggi pada fitur *Location* adalah *Lokasi Sathorn* dan *Ekkamai*.

#### Numerical Feature: 

![Numerical Feature terhadap price](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/ece9cd7d-50fa-486f-9c1f-60d552524a4a)

Gambar 9. *Numerical Feature* terhadap *Price*


Berdasarkan Gambar 9 di atas, dapat disimpulkan bahwa fitur-fitur tersebut memiliki pengaruh terhadap kenaikan harga properti.

#### Correlation Matrix: 

![Correlation Matrix](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/017763df-c6ef-4756-8454-81fb965d4895)

Gambar 10. *Correlation Matrix*

Berdasarkan Gambar 10, *correlation matrix*, dapat disimpulkan bahwa variabel "*Price*" memiliki korelasi yang cukup kuat dengan "*Area*", "*Bedrooms*", dan "*Bathrooms*". Oleh karena itu, ketiga fitur tersebut dapat menjadi kandidat yang baik untuk teknik reduksi dimensi.

## Data Preparation

Pada bagian Data Preparation, terdapat beberapa langkah preprocessing yang dilakukan secara lebih rinci. Berikut adalah langkah-langkah yang dilakukan:

1. Encoding fitur kategori menggunakan one-hot encoding:

   - Teknik yang digunakan pada tahap ini adalah *one-hot encoding*, yang digunakan untuk mengubah fitur kategori menjadi representasi numerik yang dapat digunakan oleh model.

   - Setelah dilakukan encoding pada fitur kategori, hasilnya akan menghasilkan kolom-kolom baru yang mewakili nilai-nilai kategori pada setiap fitur.

   - Misalnya, fitur Property Type memiliki kategori Apartment, Condo, dan House. Setelah dilakukan encoding, akan ditambahkan kolom-kolom baru seperti Property Type_Apartment, Property Type_Condo, dan Property Type_House. Nilai pada kolom-kolom tersebut akan menjadi 1 jika properti termasuk dalam kategori tersebut, dan 0 jika tidak.

   - Tabel 3 dibawah adalah Hasil Encoding Fitur Kategori yang telah diberikan sebelumnya memberikan contoh hasil encoding menggunakan one-hot encoding.

     Tabel 3. Hasil Encoding Fitur Kategori

     |      | Area (sq. ft.) | Bedrooms | Bathrooms | Price (THB) | Property Type_Apartment | Property Type_Condo | Property Type_House | Location_Ari | Location_Ekkamai | Location_Ladprao | Location_Phrom Phong | Location_Ratchada | Location_Sathorn | Location_Siam | Location_Silom | Location_Sukhumvit | Location_Thonglor |
     | ---- | -------------- | -------- | --------- | ----------- | ----------------------- | ------------------- | ------------------- | ------------ | ---------------- | ---------------- | -------------------- | ----------------- | ---------------- | ------------- | -------------- | ------------------ | ----------------- |
     | 0    | 700            | 1        | 1         | 2000000     | 0                       | 1                   | 0                   | 0            | 0                | 0                | 0                    | 0                 | 0                | 0             | 0              | 1                  | 0                 |
     | 1    | 1500           | 3        | 2         | 5000000     | 0                       | 0                   | 1                   | 0            | 0                | 1                | 0                    | 0                 | 0                | 0             | 0              | 0                  | 0                 |
     | 2    | 900            | 2        | 1         | 3500000     | 1                       | 0                   | 0                   | 0            | 0                | 0                | 0                    | 0                 | 0                | 1             | 0              | 0                  | 0                 |
     | 3    | 1200           | 2        | 2         | 4500000     | 0                       | 1                   | 0                   | 0            | 0                | 0                | 0                    | 0                 | 1                | 0             | 0              | 0                  | 0                 |
     | 4    | 1800           | 4        | 3         | 8000000     | 0                       | 0                   | 1                   | 0            | 0                | 0                | 0                    | 1                 | 0                | 0             | 0              | 0                  | 0                 |

2. Reduksi dimensi menggunakan Principal Component Analysis (PCA):

   - Karena terdapat korelasi yang tinggi antara fitur Area, Bathroom, dan Bedroom, dilakukan reduksi dimensi menggunakan metode *Principal Component Analysis* (PCA).

   - PCA merupakan teknik statistik yang digunakan untuk mengurangi dimensi fitur dalam dataset, sambil mempertahankan sebagian besar informasi yang terkandung dalam data.

   - Dalam langkah ini, fitur-fitur yang berkorelasi tinggi akan digabungkan menjadi komponen utama yang baru.

   - Gambar 11 dibawah adalah gambar Reduksi dimensi dengan Principal Component Analysis (PCA) yang telah diberikan sebelumnya merupakan contoh hasil reduksi dimensi menggunakan PCA. Pada gambar tersebut, dapat dilihat bahwa fitur-fitur awal (Area, Bathroom, dan Bedroom) telah direduksi menjadi dua komponen utama yang baru.

     ![Reduksi dimensi dengan Principal Component Analysis (PCA)](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/112d1c7e-e4cc-4df1-866f-1891219d9b69)

     Gambar 11. Reduksi dimensi dengan *Principal Component Analysis* (PCA).

     

3. Pembagian dataset menggunakan fungsi `train_test_split` dari library sklearn:

   - Pada tahap ini, dataset akan dibagi menjadi subset data latih (training set) dan subset data uji (testing set) menggunakan fungsi `train_test_split` dari library sklearn.

   - Proporsi pembagian dataset yang digunakan adalah 90% data untuk latihan (training set) dan 10% data untuk pengujian (testing set).

   - Hasil dari pembagian dataset dapat dilihat pada Gambar 12. Hasil pembagian menggunakan fungsi `train_test_split` dari library sklearn yang telah diberikan sebelumnya. Pada gambar tersebut, dapat dilihat bahwa dataset telah terbagi menjadi dua subset dengan proporsi yang telah ditentukan.

     ![Hasil Pembagian train_test_split](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/182a0c73-de93-442b-8498-cdcec94c99a4)

     Gambar 12. Hasil pembagian menggunakan fungsi train_test_split dari library sklearn

Dengan melakukan langkah-langkah preprocessing di atas, data telah siap untuk dilakukan tahap modelling.

## Modeling

Pada tahap pemodelan, digunakan empat algoritma modeling yaitu K-Nearest Neighbors (KNN), Random Forest, dan Boosting Algorithm. Algoritma-algoritma ini akan digunakan untuk mengembangkan model prediksi harga properti berdasarkan fitur-fitur yang ada.

### K-Nearest Neighbor

KNN (K-Nearest Neighbors) adalah salah satu algoritma dalam machine learning yang dapat digunakan untuk melakukan tugas klasifikasi dan regresi.

Keuntungan dari penggunaan KNN adalah kemudahannya dalam implementasi dan interpretasi. Algoritma ini non-parametrik, yang berarti tidak mengasumsikan distribusi tertentu pada data. KNN juga dapat mengatasi data yang kompleks atau tidak linear dan dapat memberikan hasil yang baik untuk dataset dengan jumlah sampel yang kecil.

Namun, KNN juga memiliki beberapa kelemahan. Salah satunya adalah komputasi yang intensif karena harus menghitung jarak antara setiap pasangan data dalam set pelatihan. Hal ini dapat mempengaruhi waktu komputasi ketika dataset menjadi besar. Selain itu, KNN sensitif terhadap skala data, sehingga perlu dilakukan penskalaan data yang tepat sebelumnya. Selain itu, pemilihan parameter k (jumlah tetangga) dalam KNN juga perlu diperhatikan karena dapat mempengaruhi kinerja dan generalisasi model.

KNN cocok digunakan untuk masalah klasifikasi dengan dataset yang relatif kecil dan dengan fitur-fitur yang saling terkait erat. Namun, untuk masalah regresi dan dataset dengan dimensi tinggi, KNN mungkin tidak selalu menjadi pilihan terbaik dan algoritma lain seperti regresi linear atau metode ensemble seperti random forest dapat memberikan hasil yang lebih baik. Penting untuk mengevaluasi berbagai algoritma dan mempertimbangkan karakteristik data serta tujuan yang ingin dicapai sebelum memilih algoritma yang paling sesuai untuk pemodelan.

### Random Forest

*Random Forest* adalah algoritma dalam *machine learning* yang digunakan untuk tugas klasifikasi dan regresi. Algoritma ini berbasis pada konsep *ensemble learning*, di mana multiple decision trees (pohon keputusan) digabungkan untuk membentuk "hutan" yang lebih kuat.

Setiap pohon keputusan dalam *Random Forest* dibangun secara independen menggunakan subset acak dari data pelatihan dan subset acak dari fitur yang tersedia. Proses ini disebut *bootstrap aggregating* atau *bagging*. Kemudian, prediksi dari setiap pohon digunakan untuk menentukan prediksi akhir dengan cara voting (untuk klasifikasi) atau rata-rata (untuk regresi) [2].

Keuntungan utama dari *Random Forest* adalah sebagai berikut:

1. Mengatasi masalah overfitting: Dengan menggunakan multiple decision trees yang berbeda, *Random Forest* dapat mengurangi risiko overfitting pada data pelatihan.
2. Stabilitas dan keandalan: *Random Forest* dapat mengatasi noise dan outlier dalam data, serta mampu menangani data yang tidak seimbang.
3. Mengukur pentingnya fitur: Algoritma ini dapat memberikan informasi tentang kepentingan relatif dari setiap fitur dalam membuat prediksi.

Beberapa kelemahan dari *Random Forest* adalah:

1. Komputasi yang lebih intensif: Karena melibatkan multiple decision trees, *Random Forest* dapat membutuhkan lebih banyak waktu dan sumber daya komputasi.
2. Interpretasi yang kompleks: Menginterpretasikan hasil dari *Random Forest* bisa lebih sulit daripada hanya menggunakan satu pohon keputusan tunggal.

*Random Forest* biasanya digunakan dalam berbagai jenis masalah, termasuk klasifikasi, regresi, dan deteksi anomali. Algoritma ini efektif untuk dataset dengan banyak fitur dan dataset yang cukup besar.

### Boosting Algorithm

*Boosting* adalah teknik dalam *machine learning* yang digunakan untuk meningkatkan performa model prediksi dengan menggabungkan beberapa model lemah menjadi satu model yang lebih kuat. Pada dasarnya, algoritma *boosting* berfokus pada meningkatkan kemampuan prediksi dengan mengurangi bias dan varian model.

Keuntungan dari algoritma *boosting* adalah kemampuannya menghasilkan model prediksi yang lebih kuat dibandingkan dengan model tunggal. Algoritma ini mampu menangani data yang kompleks dan memiliki performa yang baik dalam berbagai tugas seperti klasifikasi dan regresi. Namun, algoritma *boosting* juga cenderung lebih kompleks dan membutuhkan waktu komputasi yang lebih lama dibandingkan dengan algoritma lainnya.

## Evaluation

Untuk mengukur seberapa baik performa model pada masalah regresi, memang umumnya menggunakan metrik *mean squared error* (MSE). MSE mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. Semakin rendah nilai MSE, semakin baik performa model dalam memprediksi nilai target.

### Mean Squared Error (MSE)

Mean squared error (MSE) adalah salah satu ukuran kesalahan yang digunakan dalam statistik dan machine learning untuk mengevaluasi seberapa baik model regresi memprediksi nilai target. MSE menghitung rata-rata kuadrat perbedaan antara nilai prediksi dan nilai sebenarnya dari variabel target.

Secara matematis, MSE dihitung dengan mengambil selisih antara nilai sebenarnya dan nilai prediksi, mengkuadratkannya, menjumlahkan hasilnya, dan kemudian membagi jumlah ini dengan jumlah sampel. Lebih formalnya, jika kita memiliki n sampel dengan nilai sebenarnya *y_i* dan nilai prediksi *ŷ_i*, maka MSE dapat dihitung sebagai berikut:



$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 $$

Keterangan:

- **MSE**: Mean Squared Error (Rata-rata Kesalahan Kuadrat)

- **n**: Jumlah sampel dalam dataset

- **yᵢ**: Nilai sebenarnya dari target pada sampel ke-i

- **ȳ**: Nilai rata-rata dari target pada seluruh sampel

  

Semakin kecil nilai MSE, semakin baik model regresi tersebut memprediksi nilai target. Namun, MSE tidak selalu cocok untuk digunakan dalam semua situasi, misalnya ketika terdapat banyak outlier dalam data.

Hasil dari MSE menunjukan:

![Hasil dari MSE](https://github.com/mauluddin12z/Proyek-1-Machine-Learning-Dicoding/assets/71598808/c97a792f-5db2-4994-b844-847b237feffb)

Gambar 13. Hasil dari MSE
Dari Gambar 13 di atas, dapat disimpulkan bahwa *RF* adalah teknik *modeling* yang terbaik karena memberikan nilai *error* yang paling sedikit.

Berikut adalah hasil ujinya.



Tabel 4. Hasil Uji

|      | y_true  | prediksi_KNN | prediksi_RF | prediksi_soosting |
| ---- | ------- | ------------ | ----------- | ----------------- |
| 267  | 7500000 | 87500000     | 7500000     | 2037037           |

Dari tabel 4 diatas dapat dilihat, nilai Prediksi RF adalah nilai yang mendekati atau bahkan sama dengan nilai asli.

## Kesimpulan

Dalam proyek ini, dilakukan analisis prediksi harga properti menggunakan teknik Machine Learning. Beberapa temuan utama dan saran yang dihasilkan dari proyek ini adalah sebagai berikut:

1. Temuan Utama:
   - Fitur-fitur seperti ukuran properti, lokasi, jumlah kamar tidur, dan jumlah kamar mandi memiliki pengaruh signifikan terhadap harga properti.
   - Model-machine learning seperti K-Nearest Neighbors, Random Forest, dan Boosting Algorithm dapat digunakan untuk memprediksi harga properti dengan tingkat akurasi yang baik.
   - Proses preprocessing seperti encoding fitur kategori dan reduksi dimensi menggunakan PCA dapat meningkatkan performa model.
2. Saran:
   - Dalam pengembangan proyek ini di masa depan, dapat dilakukan eksplorasi lebih lanjut terhadap fitur-fitur lain yang berpotensi mempengaruhi harga properti, seperti fasilitas yang tersedia, tahun pembangunan, atau tipe properti.
   - Penambahan lebih banyak data properti dengan variasi yang lebih luas dapat meningkatkan keakuratan model prediksi.
   - Dilakukan analisis lebih mendalam terhadap faktor-faktor ekonomi, seperti suku bunga atau kondisi pasar properti, yang juga dapat mempengaruhi harga properti.
   - Mempertimbangkan penggunaan teknik lain dalam machine learning, seperti Regresi Linear atau Neural Network, untuk membandingkan performa dengan model yang telah digunakan.

## Refrensi

[1]   S. Suakanto, A. Christy, V. J. L. Engel, and D. Angela, “Pengembangan Sistem Prediksi Harga Pasar Properti Menggunakan Big Data Platform,” *J. Telemat.*, vol. 13, no. 1, pp. 19–26, 2018, [Online]. Available: https://journal.ithb.ac.id/telematika/article/view/257

[2]   G. N. Ayuni and D. Fitrianah, “Penerapan metode Regresi Linear untuk prediksi penjualan properti pada PT XYZ,” *J. Telemat.*, vol. 14, no. 2, pp. 79–86, 2019, [Online]. Available: https://journal.ithb.ac.id/telematika/article/view/321

[3]   C. Haryanto, N. Rahaningsih, F. M. Basysyar, K. Cirebon, R. F. Regression, and H. Rumah, “KOMPARASI ALGORITMA MACHINE LEARNING DALAM MEMPREDIKSI HARGA RUMAH,” vol. 1, no. 1, pp. 533–539, 2023.
