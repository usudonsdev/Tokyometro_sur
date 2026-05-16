import cv2
import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 特徴量1: 彩度（低いほどシック・上品 [2]）
    avg_sat = np.mean(hsv[:, :, 1])
    
    # 特徴量2: エッジ密度（低いほど見通しが良く機能的 [1]）
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # 特徴量3: 輝度の標準偏差（高いほど光の演出・コントラストが強い [2]）
    brightness_std = np.std(gray)
    
    return [avg_sat, edge_density, brightness_std]

LINE_COLORS = {
    "G": "#F6AE1A",
    "M": "#E60012",
    "H": "#B5B5B6",
    "T": "#009BBF",
    "C": "#00BB85",
    "Y": "#C1A470",
    "Z": "#8F76D6",
    "N": "#00ADA9",
    "F": "#9C5E31"
}

def compare_lines(line_folders):
    all_data = []
    
    for line_name, folder_path in line_folders.items():
        files = glob.glob(os.path.join(folder_path, "*.jpg"))
        for f in files:
            features = extract_features(f)
            if features:
                station = os.path.splitext(os.path.basename(f))[0]
                all_data.append([line_name, station] + features)
    
    df = pd.DataFrame(
        all_data,
        columns=['Line', 'Station', 'Saturation', 'EdgeDensity', 'BrightnessStd']
    )
    
    # --- 可視化 ---
    plt.figure(figsize=(18, 5))
    
    line_order = list(line_folders.keys())
    palette = {k: LINE_COLORS.get(k, "#666666") for k in line_order}

    # エッジ密度の比較（機能的かどうかの指標）
    plt.subplot(1, 3, 1)
    sns.boxplot(x='Line', y='EdgeDensity', data=df, order=line_order, palette=palette)
    plt.title("Comparison of Structural Simplicity (Edge Density)")
    plt.xlabel("Line")
    plt.ylabel("Edge Density")
    
    # 彩度 vs 輝度分散の散布図（上品さ・高級感のポジショニング）
    plt.subplot(1, 3, 2)
    sns.scatterplot(
        x='Saturation',
        y='BrightnessStd',
        hue='Line',
        data=df,
        s=100,
        hue_order=line_order,
        palette=palette
    )
    plt.xlabel("Saturation")
    plt.ylabel("Brightness Std")

    for i in range(df.shape[0]):
        label = df.Station.iloc[i].split('_')
        plt.text(
            x=df.Saturation[i] + 0.5, # 重ならないよう少し右に
            y=df.BrightnessStd[i], 
            s=label[0],          # H09やY16などの駅名
            fontsize=9,
            alpha=0.8
        )
        
    plt.title("Vibe Positioning: Saturation vs Light Contrast")

    # --- 路線ごとの重心プロット ---
    centroids = df.groupby('Line', as_index=False)[['Saturation', 'BrightnessStd']].mean()

    plt.subplot(1, 3, 3)
    sns.scatterplot(
        x='Saturation',
        y='BrightnessStd',
        hue='Line',
        data=centroids,
        s=140,
        hue_order=line_order,
        palette=palette,
        legend=True
    )

    for i in range(centroids.shape[0]):
        plt.text(
            x=centroids.Saturation[i] + 0.5,
            y=centroids.BrightnessStd[i],
            s=centroids.Line[i],
            fontsize=9,
            alpha=0.9
        )

    plt.title("Line Centroids: Saturation vs Brightness Std")
    plt.xlabel("Saturation")
    plt.ylabel("Brightness Std")
    plt.legend(title="Line")
    plt.tight_layout()
    plt.show()

    # --- 重心間距離の可視化（3特徴量） ---
    centroids3 = df.groupby('Line', as_index=False)[
        ['Saturation', 'EdgeDensity', 'BrightnessStd']
    ].mean()
    features = centroids3[['Saturation', 'EdgeDensity', 'BrightnessStd']].to_numpy()
    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        dist,
        xticklabels=centroids3['Line'],
        yticklabels=centroids3['Line'],
        cmap='viridis',
        annot=True,
        fmt='.2f'
    )
    plt.title("Centroid Distance (Euclidean, 3 Features)")
    plt.xlabel("Line")
    plt.ylabel("Line")
    plt.tight_layout()
    plt.show()
    return df

# 使用例: フォルダ構造を用意して実行
line_folders = {
    "H": "./img/hibiya_line/",
    "Y": "./img/yurakucho_line/",
    "G": "./img/ginza_line/",
    "Z": "./img/hanzomon_line/",
    "F": "./img/fukutoshin_line/",
    "C": "./img/chiyoda_line/",
    "N": "./img/nanboku_line/",
    "M": "./img/marunouchi_line/",
    "T": "./img/tozai_line/"
}
df_results = compare_lines(line_folders)