# Pj1: Perception System Concept and Design

Natasyahratu Zulharni | 4222401005 | RET503 Computer Vision and Deep Learning

Sistem persepsi untuk Security Patrol Robot yang mendeteksi **mobil (car)** dan **motor (motorcycle)** dengan YOLO26n, dari rancangan sampai model yang terukur. Alur yang sama dengan P3 dan P4, tetapi memakai dataset foto kelompok, bukan subset COCO.

## 1. Lingkup persepsi

Robot patroli perlu membedakan mobil dan motor di lingkungan sekitar saat berpatroli. Keluaran yang dibutuhkan sistem: kelas objek (car atau motorcycle) dan kotak pembatas di dalam gambar kamera.

## 2. Target kinerja

Isi dari dokumen desain kelompok (minggu 3). Ganti tanda `[ISI]`:

| Ukuran | Target | Hasil terukur |
|---|---|---|
| mAP50-95 (test) | [ISI] | 0,911 |
| FPS di perangkat robot | [ISI] | belum diukur di perangkat robot |
| Latensi per gambar | [ISI] ms | lihat catatan di bagian 5 |

## 3. Rancangan pipeline

Kamera: [ISI jenis dan resolusi]. Posisi pemasangan: [ISI]. Perangkat komputasi: [ISI].

```
Kamera -> Praproses (resize ke imgsz) -> YOLO26n -> NMS dan ambang kepercayaan -> Kelas dan kotak -> [ISI keluaran ke sistem robot]
```

## 4. Dataset

- Sumber: 114 foto mobil dan motor dari internet (disiapkan oleh anggota kelompok). 103 foto dipakai karena memiliki kotak objek, 11 foto dilewati karena tidak berisi objek yang terdeteksi.
- Anotasi: label awal dibuat **otomatis** dengan model YOLO pretrained COCO, lalu diperiksa dengan `01_cek_anotasi.py` (0 masalah ditemukan).
- Jumlah kotak: car 53, motorcycle 81.
- Pembagian (`05_split_dataset_yolo.py`, seed 42): train 72, val 20, test 11 gambar.
- Sidik jari label: `B423D13CAF1C`.

## 5. Eksperimen dan log

Pengaturan sama untuk ketiganya: YOLO26n, 30 epoch, batch 16, seed sama. Dijalankan di Google Colab (GPU Tesla T4) dengan `06_latih_bandingkan.py`.

| Eksperimen | Perubahan | Dataset | mAP50 | mAP50-95 | P | R | ms/gambar | Waktu latih | Kesimpulan |
|---|---|---|---|---|---|---|---|---|---|
| E1 pretrained | baseline, `imgsz=416` | dataset_pj1 | 0,995 | 0,979 | 0,990 | 1,000 | 14,3 | 1,2 menit | Baseline sangat tinggi sejak epoch pertama |
| E2 scratch | bobot acak | dataset_pj1 | 0,125 | 0,038 | 0,237 | 0,135 | 5,4 | 1,1 menit | Belum belajar dalam 30 epoch |
| E3 tuning | `imgsz=640` | dataset_pj1 | 0,995 | 0,985 | 0,988 | 1,000 | 14,0 | 1,3 menit | Setara E1 |

Model terpilih: **E3** (mAP50-95 val tertinggi). Hasil di data test (11 gambar, 19 kotak, dipakai sekali): **mAP50 0,945, mAP50-95 0,911, precision 0,822, recall 0,978**.

Hasil E3 per kelas pada data validasi:

| Kelas | Gambar | Kotak | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| car | 6 | 6 | 0,995 | 1,000 | 0,995 | 0,995 |
| motorcycle | 14 | 15 | 0,981 | 1,000 | 0,995 | 0,975 |

## 6. Analisis

**Transfer learning terbukti penting.** E1 mencapai mAP50-95 0,979, sedangkan E2 hanya 0,038 dengan data dan epoch yang sama. Dengan 72 gambar latih, model dari nol baru mulai belajar pada epoch terakhir.

**E1 dan E3 tidak bisa dibedakan.** Selisih 0,979 dan 0,985 sangat kecil, dan data validasi hanya 20 gambar dengan 21 kotak, sehingga nilainya sudah mendekati batas atas. Kesimpulannya: pada data ini `imgsz=416` dan `imgsz=640` setara, bukan bahwa 640 lebih baik.

**Angka tinggi harus dibaca dengan hati-hati.**
- Label dibuat otomatis oleh model yang berasal dari COCO, sedangkan model yang kami latih juga berasal dari COCO. Yang terukur sebagian besar adalah kesesuaian dengan pemberi label otomatis, bukan ketepatan terhadap kenyataan.
- Pada epoch pertama E1 sudah mencapai mAP50-95 0,979. Ini menunjukkan tugasnya sangat mudah bagi bobot pretrained dengan data sebersih ini.
- Foto berasal dari internet, bukan dari kamera robot, sehingga belum mewakili kondisi sebenarnya (sudut, jarak, pencahayaan, blur gerak).

**Val vs test.** mAP50-95 turun dari 0,985 (val) ke 0,911 (test), dan precision di test hanya 0,822 padahal recall 0,978. Artinya ada deteksi yang dianggap salah. Dugaan: model mendeteksi objek yang benar-benar ada tetapi tidak diberi label oleh pelabel otomatis. Dugaan ini perlu dicek dengan memeriksa gambar test satu per satu. Dengan hanya 11 gambar, angka ini juga sangat tidak stabil.

**Catatan latensi.** Waktu inferensi E3 (640) tercatat 14,0 ms dan E1 (416) 14,3 ms, padahal 640 seharusnya lebih lambat. Pengukuran dilakukan di GPU T4 Colab dengan data sangat kecil dan mengandung noise. Kecepatan untuk target FPS harus diukur ulang pada perangkat robot sebenarnya.

## 7. Keterbatasan

- Data val 20 gambar dan test 11 gambar: selisih kecil antar-eksperimen tidak bermakna.
- Satu kali run per eksperimen (satu seed).
- Label otomatis, belum diverifikasi manual seluruhnya.
- Dataset kecil (103 gambar) dan bukan dari kamera robot.
- FPS dan latensi di perangkat robot belum diukur.

## 8. Langkah lanjutan

1. Verifikasi dan koreksi label secara manual, terutama gambar test.
2. Tambah foto dari kamera robot dengan variasi sudut, jarak, dan cahaya.
3. Ukur FPS dan latensi di perangkat robot dan bandingkan dengan target bagian 2.
4. Ulangi eksperimen dengan beberapa seed supaya selisih antar-eksperimen bisa dinilai.

## 9. Isi folder

| Isi | Lokasi |
|---|---|
| Dataset berlabel YOLO | `dataset_yolo/` (contoh gambar berkotak di `dataset_yolo/cek_visual/`) |
| Dataset siap latih (train/val/test dan `data.yaml`) | `dataset_pj1/` |
| Tabel hasil | `hasil_praktik_4222401005.txt` dan `.csv` |
| Grafik pelatihan (`results.png`) | `hasil_praktik/E1_pretrained/`, `E2_scratch/`, `E3_tuning/` |

Program yang dipakai ada di folder `P3/` (data) dan `P4/` (pelatihan) di repositori ini.

## 10. Cara menjalankan ulang

```
python P3/01_cek_anotasi.py Pj1/dataset_yolo
python P3/05_split_dataset_yolo.py Pj1/dataset_yolo Pj1/dataset_pj1
python P4/06_latih_bandingkan.py --nim 4222401005 --nama "Natasyahratu Zulharni" --data Pj1/dataset_pj1/data.yaml --device 0
```
