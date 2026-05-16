import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_station_vibe(image_path):
    # 1. 画像の読み込み（信号の入力）
    img = cv2.imread(image_path)
    if img is None:
        return "画像が見つかりません"
    
    # 画像サイズを統一（正規化）
    img = cv2.resize(img, (800, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 色彩信号の解析（ラインカラーや素材感の抽出）
    # 例：日比谷線のシルバーホワイトや、銀座線のレモンイエローの占有率を計算
    # ここでは簡易的に「彩度」と「明度」の平均を算出
    avg_saturation = np.mean(hsv[:, :, 1])  # 彩度：低いほど落ち着いた・上品な印象
    avg_brightness = np.mean(hsv[:, :, 2])  # 明度：光の演出などに関連

    # 3. 構造信号の解析（エッジ密度による「機能美」の定量化）
    # シンプルな直結構造は、視覚的ノイズ（エッジ）が少ないと仮定
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size  # 全画素に対するエッジの割合

    # 4. 輝度信号の解析（光のグラデーションによる「高級感」の定量化）
    # 「上品・優雅」な間接照明は、輝度の分散（ばらつき）に現れる
    brightness_std = np.std(gray)  # 輝度の標準偏差：コントラストの強さ

    # 結果の表示
    print(f"--- 解析結果: {image_path} ---")
    print(f"【色彩・彩度】: {avg_saturation:.2f} (低いほどシック/上品)")
    print(f"【構造・エッジ密度】: {edge_density:.4f} (低いほどシンプル/機能的)")
    print(f"【光・輝度分散】: {brightness_std:.2f} (グラデーションの質に関与)")

    # 可視化
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.title("Original")
    plt.subplot(1, 2, 2); plt.imshow(edges, cmap='gray'); plt.title("Edges (Structure)")
    plt.show()



# 実行例
analyze_station_vibe('img/H09_Ginza.jpg')    # 日比谷線銀座駅（シルバーホワイト・木目）
analyze_station_vibe('img/Y16_Nagatacho.jpg')   # 有楽町線永田町駅（高級感）