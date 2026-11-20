
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# 1. Membaca dataset
df = pd.read_csv("kos_madiun.csv")

# 2. Mengubah fasilitas AC menjadi angka
df["ac_num"] = df["ac"].map({"ya": 1, "tidak": 0})

# 3. Menentukan data input (X) dan target (y)
X = df[["jarak_km", "rating", "ac_num"]]
y = df["harga"]

# 4. Membuat dan melatih model
model = LinearRegression()
model.fit(X, y)

# 5. Menyimpan model
joblib.dump(model, "model_kos.joblib")

print("Model berhasil dilatih!")
print("Model berhasil disimpan!")
print("Koefisien:", model.coef_.round(2))
