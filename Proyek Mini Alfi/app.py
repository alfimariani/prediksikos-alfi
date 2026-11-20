import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq


# =========================================================
# 1. PENGATURAN HALAMAN
# =========================================================
st.set_page_config(
    page_title="Dashboard Prediksi Harga Kos Madiun",
    page_icon="",
    layout="wide"
    
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "kos_madiun.csv"
MODEL_PATH = BASE_DIR / "model_kos.joblib"

# Cadangan jika nama file CSV di folder berbeda.
if not DATA_PATH.exists():
    alternatif_csv = BASE_DIR / "kos_madiun(1).csv"
    if alternatif_csv.exists():
        DATA_PATH = alternatif_csv


# =========================================================
# 2. DESAIN DASHBOARD
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --pink: #D96F9D;
    --pink-dark: #B95783;
    --ink: #4D4050;
    --muted: #8A7B88;
    --line: #F0DCE6;
}

.stApp {
    background: #F5F5F5;
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.3rem;
    padding-bottom: 2.5rem;
}
h1, h2, h3, h4 {
    color: var(--ink);
    font-family: 'Manrope', sans-serif;
    letter-spacing: -0.4px;
}
h1 { font-size: 2rem !important; font-weight: 800 !important; }
h2 { font-weight: 750 !important; }
h3 { font-weight: 700 !important; }

div[data-testid="stMetric"] {
    background: rgba(255,255,255,.94);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 7px 22px rgba(143, 83, 112, .07);
}
div[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: var(--pink-dark) !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.92);
    border: 1px solid var(--line);
    border-radius: 19px;
    padding: 13px 15px;
    box-shadow: 0 7px 22px rgba(143, 83, 112, .055);
}
.stButton > button {
    background: #D96F9D;
    color: white;
    border: 0;
    border-radius: 11px;
    min-height: 43px;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(217,111,157,.18);
}
.stButton > button:hover {
    color: white;
    border: 0;
}
div[data-testid="stAlert"] {
    border-radius: 13px;
}
hr { border-color: #EEDCE5; }

.kos-hero {
    background: #D96F9D;
    border-radius: 23px;
    padding: 24px 29px;
    color: white;
    margin: 2px 0 20px 0;
    box-shadow: 0 8px 20px rgba(190, 98, 139, .12);
}
.kos-hero h2 {
    color: white !important;
    font-size: 1.55rem !important;
    margin: 0 0 5px 0;
}
.kos-hero p {
    color: rgba(255,255,255,.94);
    margin: 0;
    font-size: .96rem;
}
.section-note {
    color: #8A7B88;
    font-size: .91rem;
    margin-top: -7px;
    margin-bottom: 13px;
}
.ai-box {
    background: #F0F2FF;
    border: 1px solid #E0E5FF;
    border-radius: 13px;
    padding: 13px 15px;
    margin-top: 9px;
    color: #4C5D91;
    line-height: 1.55;
}
.ai-title {
    color: #596FBD;
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .8px;
    margin-bottom: 5px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. FUNGSI BANTU
# =========================================================
def format_rupiah(nilai):
    return "Rp " + f"{nilai:,.0f}".replace(",", ".")


def arti_korelasi(nilai):
    if pd.isna(nilai):
        return "belum dapat dihitung"

    besar = abs(nilai)
    if besar < 0.20:
        tingkat = "sangat lemah"
    elif besar < 0.40:
        tingkat = "lemah"
    elif besar < 0.60:
        tingkat = "sedang"
    elif besar < 0.80:
        tingkat = "kuat"
    else:
        tingkat = "sangat kuat"

    if nilai > 0:
        arah = "positif"
    elif nilai < 0:
        arah = "negatif"
    else:
        arah = "tidak memiliki arah"

    return f"{arah} dan {tingkat}"


@st.cache_data(ttl=3600, show_spinner=False)
def analisis_semua_grafik_groq(api_key, ringkasan_json):
    """
    Satu kali pemanggilan Groq untuk membuat penjelasan empat grafik.
    Hasil disimpan dalam cache selama satu jam agar tidak memanggil API
    berulang kali pada setiap rerun Streamlit.
    """
    client = Groq(api_key=api_key)

    prompt = f"""
Kamu adalah asisten analisis data untuk dashboard Prediksi Harga Kos Madiun.

Gunakan ringkasan data berikut sebagai satu-satunya sumber angka:
{ringkasan_json}

Buat penjelasan singkat dalam bahasa Indonesia yang mudah dipahami mahasiswa.
Jangan mengarang angka. Jangan menyatakan korelasi sebagai sebab-akibat.

Kembalikan jawaban dalam format JSON valid dengan empat key:
- histogram: penjelasan sebaran harga
- jarak: penjelasan hubungan jarak dan harga
- rating: penjelasan hubungan rating dan harga
- ac: perbandingan harga kos dengan AC dan tanpa AC

Setiap nilai berupa 2 sampai 4 kalimat yang menjelaskan pola dan kesimpulan.
Jangan menambahkan teks di luar JSON.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah analis data yang teliti. "
                    "Jawab ringkas, jelas, dan hanya berdasarkan data."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_completion_tokens=1200,
        response_format={"type": "json_object"}
    )

    isi = response.choices[0].message.content
    hasil = json.loads(isi)

    # Pastikan semua bagian tersedia meskipun respons model tidak lengkap.
    kunci = ["histogram", "jarak", "rating", "ac"]
    return {
        nama: str(hasil.get(nama, "Penjelasan untuk grafik ini belum tersedia."))
        for nama in kunci
    }


def tampilkan_penjelasan_ai(teks):
    st.markdown(
        f"""
        <div class="ai-box">
            <div class="ai-title">PENJELASAN AI</div>
            {teks}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 4. MEMBACA DATASET DAN MODEL
# =========================================================
@st.cache_data
def baca_dataset(lokasi_file):
    return pd.read_csv(lokasi_file)


@st.cache_resource
def baca_model(lokasi_file):
    return joblib.load(lokasi_file)


if not DATA_PATH.exists():
    st.error(
        "File kos_madiun.csv tidak ditemukan. "
        "Letakkan file CSV dalam folder yang sama dengan app.py."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        "File model_kos.joblib tidak ditemukan. "
        "Letakkan file model dalam folder yang sama dengan app.py."
    )
    st.stop()

try:
    df = baca_dataset(str(DATA_PATH))
    model = baca_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Dataset atau model gagal dibaca: {e}")
    st.stop()

kolom_wajib = ["jarak_km", "harga", "ac", "rating"]
kolom_tidak_ada = [kolom for kolom in kolom_wajib if kolom not in df.columns]

if kolom_tidak_ada:
    st.error(f"Kolom berikut belum ada di dataset: {', '.join(kolom_tidak_ada)}")
    st.stop()

# Pastikan kolom angka terbaca sebagai numerik.
for kolom in ["jarak_km", "harga", "rating"]:
    df[kolom] = pd.to_numeric(df[kolom], errors="coerce")

df = df.dropna(subset=["jarak_km", "harga", "rating"]).copy()
df["ac"] = df["ac"].astype(str).str.lower().str.strip()
df["harga_rp"] = df["harga"] * 1000

if df.empty:
    st.error("Dataset tidak memiliki baris data yang bisa dianalisis.")
    st.stop()


# =========================================================
# 5. RINGKASAN STATISTIK UNTUK AI
# =========================================================
korelasi_jarak = df["jarak_km"].corr(df["harga"])
korelasi_rating = df["rating"].corr(df["harga"])
rata_ac = df.groupby("ac")["harga_rp"].mean()
jumlah_ac = df["ac"].value_counts().to_dict()

ringkasan_ai = {
    "jumlah_data": int(len(df)),
    "harga": {
        "minimum": float(df["harga_rp"].min()),
        "maksimum": float(df["harga_rp"].max()),
        "rata_rata": float(df["harga_rp"].mean()),
        "median": float(df["harga_rp"].median())
    },
    "jarak": {
        "minimum_km": float(df["jarak_km"].min()),
        "maksimum_km": float(df["jarak_km"].max()),
        "rata_rata_km": float(df["jarak_km"].mean()),
        "korelasi_jarak_harga": (
            None if pd.isna(korelasi_jarak) else float(korelasi_jarak)
        ),
        "interpretasi": arti_korelasi(korelasi_jarak)
    },
    "rating": {
        "rata_rata": float(df["rating"].mean()),
        "korelasi_rating_harga": (
            None if pd.isna(korelasi_rating) else float(korelasi_rating)
        ),
        "interpretasi": arti_korelasi(korelasi_rating)
    },
    "fasilitas_ac": {
        "rata_rata_harga_per_kategori": {
            str(k): float(v) for k, v in rata_ac.items()
        },
        "jumlah_data_per_kategori": {
            str(k): int(v) for k, v in jumlah_ac.items()
        }
    }
}


# =========================================================
# 6. HEADER DASHBOARD
# =========================================================

st.markdown(
    """
<div style="background-color:#D96F9D;
            border-radius:20px;
            padding:25px 30px;
            color:white;
            margin-bottom:22px;
            box-shadow:0 8px 20px rgba(190,98,139,0.12);">

<h2 style="color:white; font-size:27px; font-weight:800; margin:0;">
Dashboard Kos Madiun Alfi Mariani✦
</h2>
<p style="color:white; font-size:14px; margin-top:10px; margin-bottom:12px; line-height:1.6;">
Lihat ringkasan data, pahami faktor yang berkaitan dengan harga, dan coba perkiraan harga kos melalui Machine Learning.
</p>
<span style="display:inline-block; background:rgba(255,255,255,0.20); border:1px solid rgba(255,255,255,0.45); border-radius:20px; padding:6px 12px; color:white; font-size:11px; font-weight:600; letter-spacing:0.5px;">
DATA INSIGHT &nbsp; · &nbsp; MACHINE LEARNING
</span>
</div>
    """,
    unsafe_allow_html=True
)




# =========================================================
# 7. PREDIKSI HARGA DI BAGIAN ATAS
# =========================================================
st.subheader("Prediksi Harga Kos")
st.markdown(
    '<div class="section-note">Masukkan karakteristik kos untuk mendapatkan estimasi harga per bulan.</div>',
    unsafe_allow_html=True
)

kolom_input, kolom_hasil = st.columns([1.05, 0.95], gap="large")

with kolom_input:
    with st.container(border=True):
        st.markdown("### Input Data Kos")

        jarak_min = max(0.1, float(df["jarak_km"].min()))
        jarak_max = max(jarak_min + 0.1, float(df["jarak_km"].max()))
        jarak_default = min(max(1.7, jarak_min), jarak_max)

        jarak_input = st.slider(
            "Jarak ke kampus (km)",
            min_value=float(round(jarak_min, 1)),
            max_value=float(round(jarak_max, 1)),
            value=float(round(jarak_default, 1)),
            step=0.1
        )

        rating_input = st.slider(
            "Rating kos",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )

        ac_input = st.selectbox(
            "Fasilitas AC",
            options=["ya", "tidak"],
            format_func=lambda x: "Dengan AC" if x == "ya" else "Tanpa AC"
        )

        tombol_prediksi = st.button(
            "Prediksi Harga Kos",
            use_container_width=True
        )

with kolom_hasil:
    with st.container(border=True):
        st.markdown("### Hasil Prediksi")

        if tombol_prediksi:
            data_baru = pd.DataFrame({
                "jarak_km": [jarak_input],
                "rating": [rating_input],
                "ac_num": [1 if ac_input == "ya" else 0]
            })

            try:
                # Model dilatih menggunakan target harga dalam satuan ribuan rupiah.
                prediksi_ribuan = float(model.predict(data_baru)[0])
                prediksi_rupiah = prediksi_ribuan * 1000

                if prediksi_rupiah > 0:
                    st.success(
                        f"### {format_rupiah(prediksi_rupiah)}"
                        "\n\nPerkiraan harga kos per bulan."
                    )
                    st.caption(
                        f"Jarak {jarak_input:.1f} km · Rating {rating_input:.1f} · "
                        f"{'Dengan AC' if ac_input == 'ya' else 'Tanpa AC'}"
                    )
                else:
                    st.warning(
                        "Model menghasilkan perkiraan harga tidak positif. "
                        "Coba kombinasi input yang lain."
                    )
            except Exception as e:
                st.error(f"Prediksi gagal dilakukan: {e}")
        else:
            st.info(
                "Hasil prediksi akan muncul setelah kamu mengisi data "
                "dan menekan tombol."
            )


# =========================================================
# 8. RINGKASAN DATA
# =========================================================
st.divider()
st.subheader("Ringkasan Data Kos")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Total Data Kos", f"{len(df)} Kos")
with m2:
    st.metric("Rata-rata Harga", format_rupiah(df["harga_rp"].mean()))
with m3:
    st.metric("Rata-rata Jarak", f'{df["jarak_km"].mean():.2f} km')
with m4:
    st.metric("Rata-rata Rating", f'{df["rating"].mean():.2f} / 5')


# =========================================================
# 9. ANALISIS OTOMATIS DENGAN GROQ
# =========================================================
st.divider()
st.subheader("Analisis Data Kos")
st.markdown(
    '<div class="section-note">Grafik ditampilkan sebagai gambar tetap. '
    'Penjelasan AI akan dimuat otomatis di bawah setiap grafik.</div>',
    unsafe_allow_html=True
)

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = None

if groq_api_key:
    with st.spinner("AI sedang membaca data dan menyiapkan penjelasan grafik..."):
        try:
            hasil_ai = analisis_semua_grafik_groq(
                groq_api_key,
                json.dumps(ringkasan_ai, ensure_ascii=False)
            )
            pesan_ai_error = None
        except Exception as e:
            hasil_ai = {
                "histogram": "",
                "jarak": "",
                "rating": "",
                "ac": ""
            }
            pesan_ai_error = str(e)
else:
    hasil_ai = {
        "histogram": "",
        "jarak": "",
        "rating": "",
        "ac": ""
    }
    pesan_ai_error = (
        "API Key Groq belum ditemukan. Pastikan file "
        ".streamlit/secrets.toml berisi GROQ_API_KEY."
    )


def bagian_ai(nama_bagian):
    if pesan_ai_error:
        st.warning(f"Penjelasan AI belum berhasil dimuat: {pesan_ai_error}")
    else:
        tampilkan_penjelasan_ai(hasil_ai.get(nama_bagian, "Penjelasan belum tersedia."))


# =========================================================
# 10. GRAFIK HISTOGRAM DAN JARAK VS HARGA
# =========================================================
grafik_kiri, grafik_kanan = st.columns(2, gap="large")

with grafik_kiri:
    with st.container(border=True):
        st.markdown("### Sebaran Harga")
        st.markdown("**Histogram harga kos**")

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.hist(
            df["harga_rp"],
            bins=8,
            color="#7189EA",
            edgecolor="white",
            linewidth=1.2
        )
        ax1.set_xlabel("Harga per bulan (Rp)")
        ax1.set_ylabel("Jumlah kos")
        ax1.set_title("Sebaran Harga Kos")
        ax1.grid(axis="y", alpha=0.18)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        bagian_ai("histogram")

with grafik_kanan:
    with st.container(border=True):
        st.markdown("### Jarak vs Harga")
        st.markdown("**Jarak dan fasilitas AC**")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        warna_ac = {"ya": "#6686E8", "tidak": "#F0A34B"}

        for kategori in ["ya", "tidak"]:
            subset = df[df["ac"] == kategori]
            if not subset.empty:
                ax2.scatter(
                    subset["jarak_km"],
                    subset["harga_rp"],
                    color=warna_ac.get(kategori, "#9A8BCB"),
                    edgecolor="white",
                    linewidth=0.7,
                    alpha=0.9,
                    label="Dengan AC" if kategori == "ya" else "Tanpa AC"
                )

        ax2.set_xlabel("Jarak (km)")
        ax2.set_ylabel("Harga per bulan (Rp)")
        ax2.set_title("Jarak dan Harga Kos")
        ax2.grid(alpha=0.18)
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        bagian_ai("jarak")


# =========================================================
# 11. GRAFIK RATING VS HARGA DAN FASILITAS AC
# =========================================================
grafik_kiri2, grafik_kanan2 = st.columns(2, gap="large")

with grafik_kiri2:
    with st.container(border=True):
        st.markdown("### Rating vs Harga")
        st.markdown("**Hubungan rating dengan harga kos**")

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.scatter(
            df["rating"],
            df["harga_rp"],
            color="#A184C2",
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9
        )
        ax3.set_xlabel("Rating kos")
        ax3.set_ylabel("Harga per bulan (Rp)")
        ax3.set_title("Rating dan Harga Kos")
        ax3.grid(alpha=0.18)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        bagian_ai("rating")

with grafik_kanan2:
    with st.container(border=True):
        st.markdown("### Fasilitas AC")
        st.markdown("**Rata-rata harga berdasarkan AC**")

        rata_ac_grafik = df.groupby("ac")["harga_rp"].mean()
        urutan = [x for x in ["ya", "tidak"] if x in rata_ac_grafik.index]
        nilai_ac = [rata_ac_grafik[x] for x in urutan]
        label_ac = ["Dengan AC" if x == "ya" else "Tanpa AC" for x in urutan]
        warna_batang = ["#6686E8" if x == "ya" else "#F0A34B" for x in urutan]

        fig4, ax4 = plt.subplots(figsize=(7, 4))
        batang = ax4.bar(
            label_ac,
            nilai_ac,
            color=warna_batang,
            edgecolor="white",
            width=0.55
        )
        ax4.set_ylabel("Rata-rata harga per bulan (Rp)")
        ax4.set_title("Rata-rata Harga Berdasarkan AC")
        ax4.grid(axis="y", alpha=0.18)

        for bar in batang:
            tinggi = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                tinggi,
                format_rupiah(tinggi),
                ha="center",
                va="bottom",
                fontsize=8
            )

        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

        bagian_ai("ac")


# =========================================================
# 12. DATASET
# =========================================================
st.divider()
with st.expander("Lihat Data Dataset Kos"):
    kolom_tampil = [
        kolom for kolom in ["nama", "jarak_km", "harga_rp", "ac", "rating"]
        if kolom in df.columns
    ]
    tabel = df[kolom_tampil].copy()
    if "harga_rp" in tabel.columns:
        tabel = tabel.rename(columns={"harga_rp": "harga_per_bulan_rp"})
    st.dataframe(tabel, use_container_width=True, hide_index=True)


# =========================================================
# 13. FOOTER
# =========================================================
st.markdown("""
<div style="text-align:center; color:#9A8794; padding:26px 10px 8px 10px;">
    <b>Dashboard Prediksi Harga Kos Madiun</b><br>
    <span style="font-size:12px;">Mini Proyek Sistem Cerdas · 2026</span>
</div>
""", unsafe_allow_html=True)
