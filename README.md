# Computer-Vision-and-Deep-learning-PJ1
# Deteksi Mobil, Motor, dan Stop Sign dengan YOLO26n (Subset COCO)

Natasyahratu Zulharni | 4222401005 | RET503 Computer Vision and Deep Learning

Praktik terpadu "Dari COCO ke detektor YOLO": menyiapkan dataset deteksi dari subset COCO (P3), lalu melatih dan membandingkan tiga eksperimen YOLO26n (P4).

## 1. Dataset (P3)

- Sumber: COCO val2017, diambil dengan `00_ambil_subset_coco.py`.
- Tiga kelas: `car`, `motorcycle`, `stop sign`. `stop sign` sengaja dipilih karena jumlahnya jauh lebih sedikit di COCO, sehingga pengaruh ketidakseimbangan kelas terlihat di hasil per kelas.
- Gambar berobjek ditambah sekitar 10% gambar latar (tanpa objek).
- Pemeriksaan anotasi dengan `01_cek_anotasi.py`: tidak ada file label yang hilang. Ada puluhan peringatan "kotak sangat kecil" (di bawah 8 px pada satu sisi). Ini objek jauh di latar, bukan kesalahan format.
- Konversi YOLO ke COCO (`02`) dan kembali ke YOLO (`03`) berjalan tanpa kehilangan gambar maupun kotak.
- Pembagian data dengan `05_split_dataset_yolo.py` (70% train, 20% val, 10% test, seed 42): train [ISI jumlah] gambar, val 88 gambar (203 kotak), test 43 gambar (116 kotak).
- Sidik jari label dataset: `64700965567D`.

## 2. Eksperimen (P4)

Pengaturan yang sama untuk ketiganya: model YOLO26n, 30 epoch, batch 16, seed sama. Perbedaannya hanya satu faktor.

| Eksperimen | Pengaturan |
|---|---|
| E1 pretrained | Transfer learning dari bobot COCO (`yolo26n.pt`), `imgsz=416` |
| E2 scratch | Arsitektur sama, bobot acak (`yolo26n.yaml`), `imgsz=416` |
| E3 tuning | Seperti E1 dengan satu perubahan: `imgsz=640` |

Dijalankan di Google Colab (GPU Tesla T4) dengan `06_latih_bandingkan.py`.

## 3. Hasil

Nilai pada data validasi:

| Eksperimen | mAP50 | mAP50-95 | Precision | Recall | ms/gambar | Waktu latih |
|---|---|---|---|---|---|---|
| E1 pretrained | 0,527 | 0,379 | 0,561 | 0,526 | 5,8 | 2,8 menit |
| E2 scratch | 0,070 | 0,022 | 0,357 | 0,041 | 3,9 | 2,5 menit |
| E3 tuning (`imgsz=640`) | 0,592 | 0,440 | 0,683 | 0,546 | 7,4 | 3,3 menit |

Model terpilih: **E3** (mAP50-95 validasi tertinggi).
Hasil di data test (dipakai sekali): **mAP50 0,580 dan mAP50-95 0,405**.

Hasil E3 per kelas pada data validasi:

| Kelas | Gambar | Kotak | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| car | 56 | 149 | 0,669 | 0,389 | 0,473 | 0,287 |
| motorcycle | 19 | 41 | 0,498 | 0,557 | 0,537 | 0,360 |
| stop sign | 12 | 13 | 0,882 | 0,692 | 0,766 | 0,672 |

## 4. Analisis

**Transfer learning sangat membantu (E1 vs E2).** Dengan jumlah epoch dan data yang sama, E1 mencapai mAP50-95 0,379, sedangkan E2 hanya 0,022. Model dari nol belum sempat belajar mendeteksi objek dalam 30 epoch pada data sekecil ini, sedangkan E1 sudah membawa fitur visual dari COCO.

**Ukuran input yang lebih besar membantu (E1 vs E3).** Menaikkan `imgsz` dari 416 ke 640 menaikkan mAP50-95 dari 0,379 ke 0,440 dan precision dari 0,561 ke 0,683. Kemungkinan penyebabnya adalah banyaknya objek kecil di dataset, yang detailnya terlalu hilang saat gambar diperkecil ke 416. Harganya adalah inferensi yang lebih lambat (7,4 ms dibanding 5,8 ms per gambar) dan waktu latih yang sedikit lebih lama. Karena hanya satu perubahan yang diuji, kita belum bisa memisahkan pengaruh `imgsz` dari variasi acak antar-run.

**Hasil per kelas.** Kelas `car` memiliki mAP50-95 terendah (0,287) dengan recall hanya 0,389. Dugaan penyebabnya: mobil paling banyak muncul di gambar ramai dan sering berukuran kecil atau tertutup, dan puluhan peringatan "kotak sangat kecil" pada tahap pemeriksaan anotasi mendukung hal ini. Kelas `stop sign` justru tertinggi (0,672) walaupun datanya paling sedikit, kemungkinan karena bentuk dan warnanya khas serta biasanya tampak besar. Namun data validasi stop sign hanya berisi 13 kotak, sehingga angka ini tidak stabil dan jangan dibaca sebagai bukti stop sign lebih mudah dideteksi.

**Val vs test.** mAP50-95 di data test (0,405) sedikit lebih rendah daripada di val (0,440). Selisih sekecil ini wajar karena ukuran sampel kecil dan model dipilih memakai data val, jadi tidak ada tanda overfitting yang parah.

## 5. Keterbatasan

- Data validasi hanya 88 gambar dan data test 43 gambar, sehingga selisih kecil antar-eksperimen belum tentu bermakna.
- Setiap eksperimen hanya dijalankan satu kali (satu seed), tanpa pengulangan.
- Ketidakseimbangan kelas: `car` jauh lebih banyak daripada `motorcycle` dan `stop sign`.
- Banyak objek berukuran sangat kecil yang sulit dideteksi.

## 6. Isi repositori

| Isi | Lokasi |
|---|---|
| Program praktik (00-06) | `*.py` |
| Tabel hasil | `hasil_praktik_4222401005.txt` dan `.csv` |
| Grafik pelatihan (`results.png`) | `hasil_praktik/E1_pretrained/`, `hasil_praktik/E3_tuning/` (dan `E2_scratch/`) |
| Dataset siap latih | `dataset_v1/` (`data.yaml`) |
| Contoh gambar berkotak | `dataset_yolo/cek_visual/` |

## 7. Cara menjalankan ulang

```
pip install ultralytics opencv-python
python 00_ambil_subset_coco.py --kelas car motorcycle "stop sign" --jumlah 200
python 01_cek_anotasi.py dataset_yolo
python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json
python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
python 05_split_dataset_yolo.py dataset_yolo dataset_v1
python 06_latih_bandingkan.py --nim 4222401005 --nama "Natasyahratu Zulharni" --device 0
```
