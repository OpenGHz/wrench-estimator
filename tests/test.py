import matplotlib.pyplot as plt
import numpy as np
import json
from figure_export import save_publication_figure
from plot import style_axes

# 1. 原始数据与转换 (g 取 9.80665 以保证科学严谨)
g = 9.80665
with open("data/stats.json", "r") as f:
    data: dict = json.load(f)
    data = {float(k): v for k, v in data.items()}  # Convert keys to float

mass = np.array(sorted(data.keys()))
true_n = mass * g

means = np.array([data[m][0] for m in mass])
stds = np.array([data[m][1] for m in mass])
errors = means - true_n  # 纵轴：预测值 - 真值
errors_mean = np.mean(errors)
errors = errors - errors_mean  # 去除系统误差，突出随机误差

# 2. 绘图设置 (期刊级样式)
plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 1.2})
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

# 绘制带误差棒的柱状图
bars = ax.bar(
    true_n,
    errors,
    yerr=stds,
    width=1.5,
    color="#377eb8",
    edgecolor="black",
    capsize=5,
    error_kw={"elinewidth": 1.5, "capthick": 1.5},
)

# 3. 细节美化
style_axes(ax)
ax.axhline(0, color="black", lw=1, ls="--")  # 零误差参考线
ax.set_xlabel("Ground Truth Force (N)", fontweight="bold")
ax.set_ylabel("Prediction Error (N)", fontweight="bold")
ax.set_xticks(true_n)
ax.set_xticklabels([f"{x:.1f}" for x in true_n])
ax.grid(axis="y", ls=":", alpha=0.7)

plt.tight_layout()
save_path = "data/prediction_error"
save_publication_figure(fig, save_path, formats=["pdf"], dpi=300)
# plt.show()
