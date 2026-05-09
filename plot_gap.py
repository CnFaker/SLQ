import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ============================
# 1. 全局设置 (修改了字体)
# ============================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# 先设置 seaborn 主题
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=1.1)

# 【关键修改】设置字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 确保数学公式（如 ||gap||）也尽量使用类似风格
plt.rcParams['mathtext.fontset'] = 'stix' 

# 统一颜色：图片=红，文本=蓝
COLOR_PALETTE_PCA = {'Image': 'red', 'Text': 'blue'}

# ============================
# 2. PCA 绘图核心逻辑
# ============================
def draw_pca_subplot(image_features, text_features, ax, title):
    # 1. 拷贝与归一化
    img_feats = image_features.copy()
    txt_feats = text_features.copy()
    # 加上 1e-12 防止除零
    img_feats /= np.linalg.norm(img_feats, axis=-1, keepdims=True) + 1e-12
    txt_feats /= np.linalg.norm(txt_feats, axis=-1, keepdims=True) + 1e-12

    # 2. 计算原始高维距离 (默认值)
    dist = np.linalg.norm(img_feats.mean(axis=0) - txt_feats.mean(axis=0))

    # 3. PCA 降维
    pca = PCA(n_components=2)
    # 联合拟合
    pca.fit(np.concatenate((img_feats, txt_feats), axis=0))
    
    pca_img = pca.transform(img_feats)
    pca_txt = pca.transform(txt_feats)



    # 5. 构造 DataFrame 用于绘图
    df_img = pd.DataFrame(pca_img, columns=['x', 'y'])
    df_img['Type'] = 'Image'
    df_txt = pd.DataFrame(pca_txt, columns=['x', 'y'])
    df_txt['Type'] = 'Text'
    df = pd.concat([df_img, df_txt], ignore_index=True)

    # 6. 绘图
    sns.scatterplot(
        data=df, x="x", y="y", hue="Type",
        ax=ax, palette=COLOR_PALETTE_PCA,
        s=40, alpha=0.6, linewidth=0
    )
    
    # 7. 更新标题：显示距离
    # 使用 Times New Roman 后，标题看起来会更像论文风格
    ax.set_title(f"{title}\n||gap||: {dist:.4f}", fontsize=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # 8. 图例设置
    # loc='upper right' 可能遮挡数据，根据需要调整
    ax.legend(loc='upper right', fontsize=18, frameon=True)

# ============================
# 3. 主程序
# ============================

# --- A. 数据加载部分 ---
print("正在加载数据...")

# 【请确保文件路径正确】
try:
    full_img = torch.load('full_coco_image_embs.pt').cpu().numpy()
    full_txt = torch.load('full_coco_text_embs.pt').cpu().numpy()
    lora_img = torch.load('lora_coco_image_embs.pt').cpu().numpy()
    lora_txt = torch.load('lora_coco_text_embs.pt').cpu().numpy()
    query_img = torch.load('query_coco_image_embs.pt').cpu().numpy()
    query_txt = torch.load('query_coco_text_embs.pt').cpu().numpy()
except FileNotFoundError as e:
    print(f"错误: 找不到数据文件。请确保 .pt 文件在当前目录下。\n详细信息: {e}")
    # 为了演示，这里生成一些假数据（如果你没有文件，代码也能跑通测试字体）
    full_img = np.random.rand(100, 512)
    full_txt = np.random.rand(100, 512)
    lora_img = np.random.rand(100, 512)
    lora_txt = np.random.rand(100, 512)
    query_img = np.random.rand(100, 512)
    query_txt = np.random.rand(100, 512)

# 封装成列表方便遍历
# 注意：这里将 "Query" 改为了 "Ours" 以符合通常论文的命名习惯，如果还需要叫 Query 请改回
datasets = [
    (full_img, full_txt, "Full FT"),
    (lora_img, lora_txt, "LoRA"),
    (query_img, query_txt, "SLQ (Ours)") 
]

# --- B. 绘制并保存 PCA ---
print("正在绘制 PCA 对比图...")
fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

for i, (img, txt, title) in enumerate(datasets):
    draw_pca_subplot(img, txt, axes1[i], title)

save_path_pca = 'compare_2_pca.png'
save_path_pdf = 'compare_2_pca.pdf'

fig1.savefig(save_path_pca, bbox_inches='tight')
fig1.savefig(save_path_pdf, bbox_inches='tight')

print(f"PCA 图像已保存: {save_path_pca} 和 {save_path_pdf}")
# plt.show() # 如果在 Notebook 中运行，取消注释