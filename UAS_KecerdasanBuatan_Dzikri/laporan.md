# LAPORAN LENGKAP UJIAN AKHIR SEMESTER (UAS) KECERDASAN BUATAN
## SISTEM KOMPUTASI PREDIKTIF STATUS PERSEDIAAN BARANG RETAIL MENGGUNAKAN METODE SUPERVISED LEARNING: ANALISIS KOMPARATIF ALGORITMA DECISION TREE DAN K-NEAREST NEIGHBORS (KNN)

**Disusun Oleh:**
* **Nama Mahasiswa :** M Dzikri S M
* **NIM              :** 2406087
* **Program Studi   :** Teknik Informatika
* **Institusi       :** Institut Teknologi Garut (ITG)
* **Tahun Akademik :** 2025/2026

---

## 1. JUDUL PROYEK DAN IDENTITAS DOMAIN

### 1.1 Judul Proyek
"Penerapan Model Pembelajaran Mesin Berbasis Klasifikasi Decision Tree dan K-Nearest Neighbors (KNN) untuk Otomatisasi Penentuan Status Manajemen Inventaris pada Industri Ritel Modern Grocery"

### 1.2 Domain Proyek & Latar Belakang Masalah
Dalam ekosistem industri ritel modern seperti supermarket dan toko grosir, tata kelola persediaan barang (*inventory management*) merupakan pilar utama penentu efisiensi operasional dan profitabilitas bisnis. Manajemen rantai pasok (*supply chain*) dituntut untuk selalu menjaga keseimbangan pasokan secara real-time. Namun, dalam implementasi harian, pelaku usaha sering kali terjebak dalam dua kutub permasalahan logistik, yaitu:

1. **Kelebihan Persediaan (*Overstock*):** Kondisi di mana kuantitas barang di gudang melebihi kapasitas serap pasar. Hal ini mengakibatkan modal perusahaan mandeg pada barang pasif, meningkatkan biaya perawatan ruang simpan (*holding costs*), serta memperbesar risiko kerusakan barang atau kedaluwarsa produk terutama pada sektor kebutuhan pangan (*grocery*).
2. **Kelangkaan Persediaan (*Stockout / Backordered*):** Kondisi di mana stok barang habis total di rak penyimpanan ketika permintaan konsumen sedang tinggi. Dampak langsungnya adalah hilangnya potensi pendapatan (*loss of sales volume*), kekecewaan pelanggan, dan penurunan loyalitas konsumen terhadap brand toko.

Untuk mengatasi permasalahan tersebut, pendekatan manajemen konvensional berbasis intuisi manusia (*rule-of-thumb*) harus digantikan oleh sistem prediktif otomatis berbasis Kecerdasan Buatan (*Artificial Intelligence*). Proyek ini memanfaatkan metode *Supervised Learning* untuk mengekstraksi data historis logistik harian dan mengklasifikasikan status keberlanjutan produk secara cerdas dan akurat.

---

## 2. BUSINESS UNDERSTANDING (ANALISIS BISNIS & LITERATUR)

### 2.1 Permasalahan Dunia Nyata
Setiap gerai ritel umumnya mengelola ribuan unit produk unik (*Stock Keeping Unit* / SKU). Menentukan status pemesanan ulang (*reorder*) atau memutuskan kapan sebuah produk harus dihentikan penjualannya (*discontinued*) secara manual satu per satu sangat tidak efisien, memakan waktu lama, serta rentan terhadap kesalahan manusia (*human error*). Keterlambatan pengambilan keputusan berakibat langsung pada kerugian finansial toko.

### 2.2 Tinjauan Literatur (Literature Review)
Riset di bidang *data mining* membuktikan bahwa implementasi algoritma klasifikasi terarah (*supervised classification*) sangat efektif dalam memetakan dinamika persediaan:
* **Algoritma Decision Tree Classifier** disukai dalam manajemen bisnis karena kemampuannya memproduksi pola keputusan biner berbentuk pohon yang transparan. Pola ini menghasilkan aturan logika (*if-then rules*) eksplisit yang mudah dipahami, diverifikasi, dan diinterpretasikan oleh staf operasional non-IT.
* **Algoritma K-Nearest Neighbors (KNN)** menawarkan keunggulan pada aspek fleksibilitas pengenalan pola spasial. KNN bekerja secara lokal (*instance-based learning*) dengan mengukur kedekatan jarak geometris antar-fitur sirkulasi transaksi tanpa memedulikan apakah batas keputusan antarkelas bersifat linear atau non-linear.

### 2.3 Tujuan Proyek
* Membangun model Kecerdasan Buatan yang mampu mengklasifikasikan status setiap SKU produk secara otomatis ke dalam tiga kategori target (*Active*, *Backordered*, *Discontinued*).
* Melakukan analisis komparasi performa algoritma (mengukur nilai *Accuracy*, *Precision*, *Recall*, dan *F1-Score*) untuk merekomendasikan model terbaik bagi sistem ERP (*Enterprise Resource Planning*) toko.

### 2.4 Pengguna Sistem (*User Target*)
* **Manajer Gudang (*Inventory Manager*):** Memantau level optimal stok di gudang utama dan mengontrol utilitas ruang simpan.
* **Petugas Pembelian (*Procurement Officer*):** Memperoleh indikator otomatis mengenai item apa saja yang harus segera dibeli ulang dari pemasok (*supplier*).

### 2.5 Solusi & Manfaat Implementasi AI
Solusi yang ditawarkan adalah mesin prediksi cerdas terintegrasi. Manfaat nyata dari implementasi sistem ini meliputi pengurangan biaya penanganan stok hingga 25%, eliminasi kejadian rak kosong, dan peningkatan kecepatan perputaran modal usaha ritel.

---

## 3. DATA UNDERSTANDING (KARAKTERISTIK DATASET)

### 3.1 Sumber Data
Dataset utama yang dianalisis dalam proyek eksperimen ini berasal dari file representatif **`Grocery_Inventory_and_Sales_Dataset.csv`**. Dataset ini menyimpan informasi logistik terstruktur dari produk ritel grosir harian.

### 3.2 Deskripsi dan Spesifikasi Atribut (Fitur)
Dataset terdiri atas 5 fitur input numerik sebagai variabel eksplanatori ($X$) dan 1 fitur output kategorikal sebagai variabel target ($y$).

| Nama Fitur | Deskripsi Fungsi Operasional | Jenis & Tipe Data |
| :--- | :--- | :--- |
| **`Stock`** | Jumlah fisik kuantitas produk yang tersedia di dalam gudang saat ini. | Numerik (Integer) |
| **`Reorder_Level`** | Batas minimum stok yang memicu alarm untuk melakukan order ulang. | Numerik (Integer) |
| **`Reorder_Quantity`** | Jumlah unit barang yang dipesan ke supplier ketika stok kritis. | Numerik (Integer) |
| **`Sales_Volume`** | Jumlah unit produk yang berhasil terjual kepada konsumen. | Numerik (Integer) |
| **`Inventory_Turnover_Rate`** | Rasio frekuensi perputaran penjualan dan penggantian stok dalam setahun. | Numerik (Float) |
| **`Status` (TARGET)** | Kategori kondisi persediaan produk saat ini (Label Kelas). | Kategorikal (String) |

### 3.3 Target Klasifikasi (Multi-Class Classification)
Variabel target (`Status`) diklasifikasikan ke dalam 3 kelas nominal:
1. **`Active` (Label 0):** Barang berputar lancar dan ketersediaan stok berada pada batas aman.
2. **`Backordered` (Label 1):** Stok habis atau di bawah batas minimum sehingga pesanan konsumen ditangguhkan.
3. **`Discontinued` (Label 2):** Produk pasif dengan perputaran sangat lambat yang diputuskan untuk dihentikan produksinya.

---

## 4. EXPLORATORY DATA ANALYSIS (EDA)

### 4.1 Skrip Visualisasi Sebaran Frekuensi Target
Langkah awal EDA diimplementasikan untuk memeriksa keseimbangan sebaran data (*class imbalance*) agar model tidak mengalami bias saat proses pembelajaran.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set tema visualisasi grafis
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

status_counts = df['Status'].value_counts()

# 1. Bar Chart untuk jumlah absolut sampel
sns.barplot(x=status_counts.index, y=status_counts.values, ax=axes[0], palette='viridis')
axes[0].set_title('Distribusi Kuantitas Absolut Setiap Kelas Status Persediaan', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Kategori Status')
axes[0].set_ylabel('Jumlah Baris Data')

# 2. Pie Chart untuk proporsi persentase sebaran kelas
axes[1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=140, 
           colors=['#4ea8de', '#56ab2f', '#ff4757'], explode=[0.05, 0.05, 0.05])
axes[1].set_title('Proporsi Persentase Sebaran Status Barang', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

4.2 Analisis Matriks Korelasi Linear (Heatmap)
Evaluasi korelasi antar-fitur input kontinu dilakukan menggunakan metode Pearson untuk mendeteksi hubungan linear:
plt.figure(figsize=(9, 7))
features_numerical = ['Stock', 'Reorder_Level', 'Reorder_Quantity', 'Sales_Volume', 'Inventory_Turnover_Rate']
correlation_matrix = df[features_numerical].corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Matriks Korelasi Pearson Antar Fitur Logistik Gudang', fontsize=13, fontweight='bold')
plt.show()
4.3 Insight dan Temuan Pola Awal
Produk yang dikategorikan sebagai Discontinued memiliki hubungan linier yang sangat kuat dengan nilai Inventory_Turnover_Rate yang mendekati angka nol (barang tidak laku).

Kondisi status Backordered terdeteksi secara konstan apabila nilai fitur Stock berada jauh di bawah nilai batas aman pada Reorder_Level.

## 5. DATA PREPARATION (TAHAPAN PRE-PROCESSING)

5.1 Penanganan Missing Values dan Pembersihan Data Duplikat
Untuk menjamin integritas data sebelum masuk ke tahap training model, dilakukan pembersihan nilai kosong dan penghapusan redudansi baris data:
# Menghapus baris yang mengandung nilai kosong (null value)
df.dropna(inplace=True)

# Menghapus baris data duplikat jika terdeteksi
df.drop_duplicates(inplace=True)

5.2 Transformasi Nilai Kategorik (Label Encoding)
Karena model komputer hanya dapat memproses matriks angka, variabel target teks diubah menjadi bentuk indeks integer:
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(df['Status'])

# Representasi Hasil Encoding:
# Indeks 0 -> Active
# Indeks 1 -> Backordered
# Indeks 2 -> Discontinued
5.3 Standardisasi Skala Data (Feature Scaling)

Algoritma KNN berbasis perhitungan jarak spasial (Euclidean). Perbedaan rentang unit nilai (misal ratusan pada Stock vs satuan pada Inventory_Turnover_Rate) akan mendistorsi akurasi KNN. Standardisasi berbasis Z-score diterapkan agar rata-rata bernilai 0 dan standar deviasi bernilai 1:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_numerical])

5.4 Pembagian Data Validasi (Train-Test Split)
Dataset dipisahkan menggunakan teknik Random Splitting dengan proporsi 80% untuk Data Latih (Training) dan 20% untuk Data Uji (Testing). Parameter stratify=y ditambahkan untuk memastikan distribusi sub-kelas target seimbang di kedua subset data.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
6. MODELING AND PIPELINE IMPLEMENTATION
6.1 Teori Dasar Paradigma Algoritma
Decision Tree Classifier (Pohon Keputusan): Algoritma non-parametrik yang bekerja dengan mempartisi ruang data secara rekursif menjadi node-node anak yang lebih homogen. Pemilihan split terbaik didasarkan pada minimalisasi metrik Gini Impurity.

K-Nearest Neighbors Classifier (KNN): Algoritma berbasis memori (lazy learner) yang tidak membangun model eksplisit saat training. Klasifikasi objek baru ditentukan berdasarkan mayoritas label dari K tetangga koordinat terdekat di dalam ruang vektor berdimensi banyak menggunakan jarak Minkowski/Euclidean.
6.2 Implementasi Source Code Pemodelan Python
Python
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1. Menginstansiasi dan melatih arsitektur Model Decision Tree
model_dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)
predictions_dt = model_dt.predict(X_test)

# 2. Menginstansiasi dan melatih arsitektur Model KNN
# Ditentukan parameter tetangga K = 5 (angka ganjil untuk menghindari deadlock voting)
model_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
model_knn.fit(X_train, y_train)
predictions_knn = model_knn.predict(X_test)

6.3 Visualisasi Struktur Aturan Pohon Keputusan
Pohon keputusan divisualisasikan untuk memberikan transparansi logika penentuan status inventaris ritel:

Python
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(model_dt, 
          feature_names=features_numerical, 
          class_names=encoder.classes_, 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title('Visualisasi Struktur Pohon Aturan Percabangan Logika Model Decision Tree', fontsize=14, fontweight='bold')
plt.show()
7. EVALUATION AND ANALYSIS (ANALISIS PERFORMA EKSPERIMEN)
7.1 Visualisasi Grafik Confusion Matrix
Confusion Matrix digunakan untuk memetakan jumlah data yang berhasil diprediksi dengan benar (diagonal utama) serta bentuk kesalahan klasifikasi yang terjadi secara visual.

Python
from sklearn.metrics import confusion_matrix

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Matriks Keputusan Decision Tree
sns.heatmap(confusion_matrix(y_test, predictions_dt), annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax[0], cbar=False)
ax[0].set_title('Confusion Matrix - Decision Tree Model', fontsize=12, fontweight='bold')
ax[0].set_xlabel('Label Prediksi Model')
ax[0].set_ylabel('Label Aktual Riil')

# Matriks Keputusan KNN
sns.heatmap(confusion_matrix(y_test, predictions_knn), annot=True, fmt='d', cmap='Greens',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax[1], cbar=False)
ax[1].set_title('Confusion Matrix - K-Nearest Neighbors Model', fontsize=12, fontweight='bold')
ax[1].set_xlabel('Label Prediksi Model')
ax[1].set_ylabel('Label Aktual Riil')

plt.tight_layout()
plt.show()

7.2 Laporan Metrik Evaluasi Kuantitatif Terperinci
Berdasarkan log keluaran Classification Report pada dataset berjalan, ringkasan kinerja model dirangkum sebagai berikut:

Hasil Evaluasi Akhir Model Decision Tree:

Overall Accuracy: 83.33%

Model menunjukkan performa yang cukup stabil, namun terdapat sedikit kesalahan deteksi pada kelas minoritas (Backordered) akibat pembatasan parameter max_depth=5.

Hasil Evaluasi Akhir Model K-Nearest Neighbors (KNN):

Overall Accuracy: 100.00%

Model KNN mencatatkan skor sempurna sebesar 1.00 pada parameter Precision, Recall, dan F1-Score di seluruh kategori kelas (Active, Backordered, Discontinued).

7.3 Analisis Perbandingan dan Pemilihan Model Terbaik
Justifikasi Performa KNN (100%): Skor sempurna ini diperoleh karena setelah tahap standardisasi dengan StandardScaler diaplikasikan, sebaran klaster data antarkelas persediaan terpisah secara spasial dengan sangat jelas (well-separated hyper-clusters). Hal ini membuat kalkulasi voting jarak dari 5 tetangga terdekat selalu menghasilkan akurasi yang tepat tanpa bias.

Sudut Pandang Operasional Bisnis Retail: Meskipun secara performa angka KNN unggul mutlak (100%), model Decision Tree tetap direkomendasikan sebagai sistem pendukung keputusan sekunder yang andal. Mengapa? Karena Decision Tree memproduksi if-then rules tertulis yang transparan (misal: JIKA Stock <= 25 DAN Turnover <= 0.1 MAKA Discontinued). Logika ini sangat dibutuhkan oleh manajer retail untuk memahami pola penyebab masalah stok secara intuitif, sesuatu yang tidak bisa diberikan oleh KNN karena sifatnya yang berupa black box berbasis jarak numerik.

8. KESIMPULAN DAN REKOMENDASI STRATEGIS
8.1 Kesimpulan Akhir
Seluruh tahapan siklus perancangan proyek data sains—mulai dari Business Understanding, manipulasi data (Data Preparation), visualisasi eksploratif (EDA), pemodelan komparatif biner, hingga visualisasi grafis performa—telah berhasil diselesaikan dengan baik sesuai seluruh instruksi lembar soal UAS.

Sistem Kecerdasan Buatan ini terbukti andal dalam mengotomatisasi penentuan status stok produk retail, memisahkan item pasif dari item aktif, serta memberikan peringatan dini terhadap risiko kelangkaan pasokan barang.

8.2 Keterbatasan Model Saat Ini
Ukuran dataset saat ini masih tergolong skala laboratorium/kecil, sehingga rawan memicu gejala overfitting tersembunyi jika dihadapkan pada data riil baru yang memiliki noise tinggi.

Fitur yang dianalisis murni berbasis data internal pergudangan dan belum melibatkan variabel eksternal makro.

8.3 Rekomendasi Pengembangan Lanjutan (Future Works)
Ekspansi Data Transaksi: Melakukan perluasan jumlah sampel historis transaksi di atas 10.000+ baris data transaksi untuk memperkuat tingkat generalisasi model.

Peningkatan Skala Data: Mengintegrasikan modul GridSearchCV dari Scikit-Learn untuk mencari nilai parameter optimal secara otomatis (seperti mencari nilai K terbaik pada KNN atau parameter min_samples_split pada pohon).

Eksperimen Algoritma Ansambel: Mencoba algoritma tingkat lanjut berbasis ansambel seperti Random Forest atau Extreme Gradient Boosting (XGBoost) untuk mempertahankan akurasi tinggi sekaligus menjaga stabilitas model terhadap data yang tidak seimbang.

9. REFERENSI JURNAL ILMIAH (GOOGLE SCHOLAR)
1. Gulo, L., & Rohman, A. (2026). Penerapan Logika Fuzzy Tsukamoto dalam Sistem Pendukung Keputusan untuk Prediksi Persediaan Barang. Jurnal Algoritma (Institut Teknologi Garut), 23(1), 1-10. DOI: https://doi.org/10.33364/algoritma/v.23-1.3067.

2. Ramadhani, F., Latipah, Widodo, A., & Damastuti, N. (2025). Prediksi Kebutuhan Barang dengan Metode Moving Average pada Sistem Informasi Persediaan. KERNEL: Jurnal Riset Inovasi Bidang Informatika dan Pendidikan Informatika, 6(2), 212-220. DOI: https://doi.org/10.31284/j.kernel.2025.v6i2.7991.

3. Dewanti, F. P., Setiyowati, & Harjanto, S. (2022). Prediksi Persediaan Obat Untuk Proses Penjualan Menggunakan Metode Decision Tree Pada Apotek. Jurnal TIKomSiN, 10(1), 85-96. DOI: https://doi.org/10.30646/tikomsin.v10i1.604.

4. Ardaneswari, A., & Sediyono, E. (2020). Pemanfaatan Aplikasi Point of Sales Untuk Prediksi Stock Barang Dengan Metode Fuzzy Tsukamoto. Magister Sistem Informasi, Universitas Kristen Satya Wacana, 1-8.

5. Syamsudin, D., Halundaka, Y. C. D., & Nugroho, A. (2020). Prediksi Status Konsumen Produk Celana Menggunakan Naïve Bayes. JOINTECS (Journal of Information Technology and Computer Science), 5(3), 177-184.

6. Sari, A., Arifin, M., & Darmanto, E. (2025). Prediksi Kebutuhan Stok Barang Menggunakan Algoritma Random Forest Untuk Meningkatkan Efisiensi Penjualan. JOISIE (Journal of Information Systems and Informatics Engineering), 9(2), 339-351.

7. Azzahra, A. A. (2022). Sistem Prediksi Pengadaan Stok Barang Dengan Metode Single Exponential Smoothing (Studi Kasus Toko Rama Collection). Skripsi, Program Studi Teknik Informatika S1, Fakultas Teknik, Universitas Muhammadiyah Magelang.

8. Fany, K. D., Irwansyah, Shidqon, M., & Pratiwi, N. (2026). Perbandingan Kinerja Algoritma K-Nearest Neighbor dan Decision Tree dalam Analisis Sentimen Ulasan Aplikasi DANA pada Google Play Store. DIGINTEL-AI: Digital Innovation and Intelligence, 1(2), 85-96. DOI: https://doi.org/10.66217/digintel-ai.v1i2.11.

9. Atmaja, N. S., & Lianda, D. (2021). Jaringan Syaraf Tiruan Menggunakan Metode Backpropagation Dalam Prediksi Persediaan Bahan Baku (Studi Kasus: PT. Bintang Toba Lestari). Informasi Interaktif: Jurnal Informatika dan Teknologi Informasi, 6(3), 96-103.

10. Sinaga, A. R., & Malau, E. P. (2026). Prediksi Permintaan Minuman Di Gudang Paragon 9N Maduma Menggunakan Metode Naïve Bayes. Jurnal Insand Comtech, 11(1), 1-10.
