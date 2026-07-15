# 🧠 UAS Kecerdasan Buatan - Prediksi Status Persediaan Barang
### 📊 Implementasi Model Decision Tree & K-Nearest Neighbors (KNN)
### 🏫 Institut Teknologi Garut (ITG) - 2025

Proyek ini merupakan implementasi dari Ujian Akhir Semester (UAS) mata kuliah Kecerdasan Buatan. Program ini bertujuan untuk melakukan analisis prediktif terhadap status persediaan barang (Active, Discontinued, Backordered) pada industri retail menggunakan dataset **Grocery Inventory and Sales Dataset**.

---

## 📌 Anggota Tim / Pembuat
* **Nama:** M Dzikri S M
* **NIM:** 2406087
* **Program Studi:** Teknik Informatika
* **Kampus:** Institut Teknologi Garut (ITG)

---

## 🚀 Fitur Utama Program
Sesuai dengan 9 poin ketentuan utama tugas, program ini mencakup:
1. **Import Library Utama:** Menggunakan pustaka standar data science (`Pandas`, `Numpy`, `Matplotlib`, `Seaborn`, `Scikit-Learn`).
2. **Pemuatan Dataset Riil:** Integrasi langsung dengan berkas CSV eksternal `Grocery_Inventory_and_Sales_Dataset.csv`.
3. **Exploratory Data Analysis (EDA):** Tampilan ringkasan statistik, distribusi data target, serta visualisasi otomatis berupa *Bar Chart* dan *Pie Chart*.
4. **Data Preparation & Splitting:** Proses transformasi kelas menggunakan `LabelEncoder` dan pembagian data secara acak teratur dengan rasio 80:20 (`test_size=0.2`).
5. **Normalisasi Data / Fitur Scaling:** Menggunakan `StandardScaler` untuk memastikan performa algoritma berbasis jarak (KNN) bekerja dengan optimal tanpa mengganggu algoritma pohon.
6. **Pemodelan Decision Tree Classifier:** Pelatihan model berbasis pohon dengan visualisasi bagan grafis pohon keputusan (`plot_tree`).
7. **Pemodelan K-Nearest Neighbors (KNN):** Pelatihan model klasifikasi berbasis kedekatan jarak spasial objek.
8. **Evaluasi Komprehensif:** Metrik performa presisi lewat grafik *Confusion Matrix Heatmap* dan *Classification Report* multi-kelas.
9. **Kesimpulan & Jurnal:** Dokumentasi hasil analisis dan pencantuman 5 referensi jurnal ilmiah akademik.

---

## 🛠️ Cara Menjalankan Program
1. Pastikan Anda telah mengunduh berkas notebook utama: `UAS_KecerdasanBuatan_Grocery_v2.ipynb`.
2. Letakkan file dataset `Grocery_Inventory_and_Sales_Dataset.csv` di dalam folder atau path yang sama (atau sesuaikan variabel `csv_path` di dalam kode).
3. Buka file notebook melalui **Google Colab** atau **Jupyter Notebook** lokal Anda.
4. Pilih menu **Runtime > Run All** atau jalankan setiap sel kode secara berurutan dari atas ke bawah.

---

## 📦 Kebutuhan Perangkat Lunak (Dependencies)
Instalasi library dapat dilakukan menggunakan perintah berikut jika belum tersedia:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
