import numpy as np
import matplotlib.pyplot as plt


# 模拟数据：x=2 对应了三个不同的 y 值 (3.0, 3.5, 2.8)
x = np.array([1, 2, 2, 2, 3, 4, 5])
y = np.array([1.1, 3.0, 3.5, 2.8, 3.9, 5.1, 5.8])

# 直接拟合，无需去重或预处理
slope, intercept = np.polyfit(x, y, 1)

print(f"拟合结果: y = {slope:.4f}x + {intercept:.4f}")

# --- 可选：可视化验证 ---
plt.scatter(x, y, label="raw data", color="red", alpha=0.6)
plt.plot(
    x,
    slope * x + intercept,
    label=f"fitted: y={slope:.2f}x+{intercept:.2f}",
    color="blue",
)
plt.legend()
plt.show()
