
# PHÂN CỤM SINH VIÊN K58KTP - K-MEANS
# Môn: Khoa học Dữ liệu

````
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── BƯỚC 1: ĐỌC & XỬ LÝ DỮ LIỆU ────────────────────────────
df_raw  = pd.read_excel("data/TỔNG HỢP ĐIỂM K58KTP.xlsx", header=None)
ten_sv  = df_raw.iloc[2, 3:].tolist()
mssv    = df_raw.iloc[1, 3:].tolist()
ten_mon = df_raw.iloc[4:, 2].tolist()

diem_num = df_raw.iloc[4:, 3:].apply(pd.to_numeric, errors='coerce')
df = diem_num.T.copy()
df.columns = ten_mon
df.insert(0, 'MSSV', mssv)
df.insert(1, 'Ten_SV', ten_sv)

X = df.drop(columns=['MSSV', 'Ten_SV'])
X = X.apply(lambda col: col.fillna(col.mean()))
df['DTB'] = X.mean(axis=1).round(3)

print(f"✅ Đọc xong: {len(df)} sinh viên, {X.shape[1]} môn học")

# ── BƯỚC 2: CHUẨN HÓA ───────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── BƯỚC 3: ELBOW METHOD ────────────────────────────────────
inertia = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(2, 8), inertia, 'o-', color='steelblue', linewidth=2)
plt.axvline(x=3, color='red', linestyle='--', label='K=3')
plt.title('Elbow Method')
plt.xlabel('So cum K')
plt.ylabel('Inertia')
plt.legend()
plt.tight_layout()
plt.savefig('elbow.png', dpi=150)
plt.show()

# ── BƯỚC 4: HUẤN LUYỆN K-MEANS K=3 ─────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df['Cum'] = labels

cum_tb = df.groupby('Cum')['DTB'].mean().sort_values()
ten_nhom = {
    cum_tb.index[0]: 'Nhom C - Trung binh/Yeu',
    cum_tb.index[1]: 'Nhom B - Kha',
    cum_tb.index[2]: 'Nhom A - Gioi/Xuat sac'
}
df['Nhom'] = df['Cum'].map(ten_nhom)

# ── BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ────────────────────────────────
sil = silhouette_score(X_scaled, labels)
print(f"\n✅ Silhouette Score: {sil:.4f}")
print("\n=== SỐ SV MỖI NHÓM ===")
print(df['Nhom'].value_counts())
print("\n=== ĐIỂM TB MỖI NHÓM ===")
print(df.groupby('Nhom')['DTB'].agg(['mean', 'min', 'max']).round(3))

# ── BƯỚC 6: BIỂU ĐỒ PCA 2D ──────────────────────────────────
pca   = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

colors = {
    'Nhom A - Gioi/Xuat sac':  '#2ecc71',
    'Nhom B - Kha':             '#f39c12',
    'Nhom C - Trung binh/Yeu': '#e74c3c'
}

plt.figure(figsize=(9, 6))
for nhom, color in colors.items():
    mask = df['Nhom'] == nhom
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=nhom, color=color, alpha=0.75, s=100)
plt.title(f'K-Means K=3 - PCA 2D (Silhouette={sil:.4f})')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_pca.png', dpi=150)
plt.show()

# ── BƯỚC 7: XUẤT EXCEL ───────────────────────────────────────
df[['MSSV', 'Ten_SV', 'DTB', 'Nhom']].sort_values('Nhom') \
  .to_excel('data/ketqua_phancum.xlsx', index=False)
print("\n✅ Xong! Kết quả lưu tại: data/ketqua_phancum.xlsx")