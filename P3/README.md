# P3: Menyiapkan Dataset Deteksi dari Subset COCO

Natasyahratu Zulharni | 4222401005 | RET503 Computer Vision and Deep Learning

Tahap 1-2 praktik terpadu "Dari COCO ke detektor YOLO": membuat dataset YOLO dari COCO, memeriksa anotasinya, menguji konversi dua arah YOLO dan COCO, lalu membaginya menjadi train, val, dan test.

## Dataset

- Sumber: COCO val2017, diambil dengan `00_ambil_subset_coco.py`.
- Tiga kelas: `car`, `motorcycle`, `stop sign`. `stop sign` sengaja dipilih karena jumlahnya jauh lebih sedikit di COCO, sehingga pengaruh ketidakseimbangan kelas terlihat pada hasil per kelas di P4.
- Gambar berobjek ditambah sekitar 10% gambar latar (tanpa objek).
- Pembagian data (`05_split_dataset_yolo.py`, seed 42, proporsi 70/20/10): train 292 gambar, val 88 gambar (203 kotak), test 43 gambar (116 kotak).
- Sidik jari label dataset: `64700965567D`.

## Langkah dan hasil

| Langkah | Program | Hasil |
|---|---|---|
| 1 | `00_ambil_subset_coco.py` | `dataset_yolo/` berisi gambar dan label YOLO untuk 3 kelas |
| 2 | `01_cek_anotasi.py` | Tidak ada file label yang hilang. Ada puluhan peringatan "kotak sangat kecil" (kurang dari 8 px pada satu sisi) |
| 3 | `02_yolo_ke_coco.py` | YOLO ke COCO JSON tanpa error |
| 4 | `03_coco_ke_yolo.py` | COCO kembali ke YOLO, jumlah file label sama dengan jumlah gambar |
| 5 | `05_split_dataset_yolo.py` | `dataset_v1/` dengan train/val/test dan `data.yaml` |

Peringatan "kotak sangat kecil" berasal dari objek jauh di latar gambar COCO, bukan kesalahan format. Ini penting untuk analisis P4 karena objek kecil sulit dideteksi.

## Isi folder

| Isi | Lokasi |
|---|---|
| Program 00, 01, 02, 03, 05 | `*.py` |
| Dataset siap latih (train/val/test dan `data.yaml`) | `dataset_v1/` |
| Contoh gambar dengan kotak label | `dataset_yolo/cek_visual/` |

Catatan: `path` di `dataset_v1/data.yaml` masih menunjuk ke lokasi di Google Colab (`/content/dataset_v1`). Untuk memakainya di komputer lain, ubah baris `path:` ke lokasi folder `dataset_v1` di komputer tersebut atau jalankan ulang `05_split_dataset_yolo.py`.

## Cara menjalankan ulang

```
pip install opencv-python
python 00_ambil_subset_coco.py --kelas car motorcycle "stop sign" --jumlah 200
python 01_cek_anotasi.py dataset_yolo
python 02_yolo_ke_coco.py dataset_yolo dataset_yolo/annotations_coco.json
python 03_coco_ke_yolo.py dataset_yolo/annotations_coco.json hasil_balik
python 05_split_dataset_yolo.py dataset_yolo dataset_v1
```

Lanjutan: pelatihan dan perbandingan model ada di folder `../P4`.
