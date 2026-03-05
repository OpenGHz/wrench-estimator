import seaborn as sns
import matplotlib.pyplot as plt


"""设置绘图风格"""
# # 启用 Seaborn 的白色网格主题
# sns.set_theme(style="whitegrid")
# 设置字体类型
plt.rcParams.update(
    {
        # 设置中文字体（防止乱码，视系统环境而定，这里主要使用英文标签以确保通用性）；
        # 同时解决负号显示问题 (有些字体不支持负号，会显示方块)
        "axes.unicode_minus": False,
        # 设置字体为 Arial 或其他常见的无衬线字体
        # Linux (如 Ubuntu)，通常没有 Arial，但预装了开源字体 DejaVu Sans
        # 因此我们可以指定一个字体列表，Matplotlib 会按顺序查找可用字体并使用第一个找到的字体
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    }
)
# 设置字体大小
# plt.rcParams.update(
#     {
#         "font.size": 14,
#         "axes.titlesize": 16,
#         "axes.labelsize": 14,
#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "legend.fontsize": 13,
#     }
# )


def style_axes(ax: plt.Axes) -> None:
    # 移除顶部和右侧的边框线（spines）
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    # 设置剩余边框线的颜色和宽度
    # for side in ("left", "bottom"):
    #     ax.spines[side].set_color("black")
    #     ax.spines[side].set_linewidth(1.2)
    # 将网格线放在数据点下方
    ax.set_axisbelow(True)
    # 将水平网格线设置为虚线，颜色为浅灰色，适当调整透明度
    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, color="0.7", alpha=0.8)
    ax.xaxis.grid(False)


if __name__ == "__main__":
    # 创建画布，分为 2x2 的四个子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Seaborn Visualization Examples", fontsize=16)
    for ax in axes.flat:
        style_axes(ax)

    # 加载示例数据
    tips = sns.load_dataset("tips")
    fmri = sns.load_dataset("fmri")

    # 1. 散点图 (Scatter Plot)
    # 展示总账单金额 (total_bill) 与小费 (tip) 之间的关系，用吸烟者 (smoker) 区分颜色
    sns.scatterplot(
        data=tips,
        x="total_bill",
        y="tip",
        s=100,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("1. Scatter Plot: Total Bill vs Tip")
    axes[0, 0].set_xlabel("Total Bill ($)")
    axes[0, 0].set_ylabel("Tip ($)")

    # 2. 箱线图 (Box Plot)
    # 展示不同星期 (day) 的总账单金额分布
    sns.boxplot(
        data=tips,
        x="day",
        y="total_bill",
        ax=axes[0, 1],
        linewidth=1.5,
    )
    axes[0, 1].set_title("2. Box Plot: Total Bill Distribution by Day & Sex")
    axes[0, 1].set_xlabel("Day of Week")
    axes[0, 1].set_ylabel("Total Bill ($)")

    # 3. 带有误差线的柱状图 (Bar Plot with Error Bars)
    # Seaborn 的 barplot 默认会显示置信区间 (CI) 作为误差线
    # 这里计算不同类别 (category) 的平均值，误差线代表 95% 置信区间
    sns.barplot(
        data=tips,
        x="day",
        y="total_bill",
        ax=axes[1, 0],
        errorbar="ci",  # 显式指定误差线为置信区间 (默认也是ci)，还可以设为 "sd" (标准差)
        capsize=0.1,  # 误差线顶部的横线长度
    )
    axes[1, 0].set_title("3. Bar Plot with Error Bars (95% CI)")
    axes[1, 0].set_xlabel("Day of Week")
    axes[1, 0].set_ylabel("Average Total Bill ($)")

    # 4. 带置信区间的折线图 (Line Plot with Confidence Interval)
    # 使用 fmri 数据集，展示时间点上信号的平均值，阴影部分代表置信区间
    sns.lineplot(
        data=fmri,
        x="timepoint",
        y="signal",
        ax=axes[1, 1],
        errorbar="ci",  # 默认就是置信区间，也可以设为 "sd" (标准差)
        marker="o",  # 添加标记点
    )
    axes[1, 1].set_title("4. Line Plot with Confidence Interval (Trend)")
    axes[1, 1].set_xlabel("Timepoint")
    axes[1, 1].set_ylabel("Signal")

    # 调整布局以防止标题重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 显示图表
    plt.show()
    # 保存图表
    from figure_export import check_figure_size, save_publication_figure
    from pathlib import Path

    check_figure_size(fig)
    path = Path(__file__).parent / "data" / "seaborn_visualization_examples"
    path.parent.mkdir(exist_ok=True)
    save_publication_figure(fig, path, formats=["png", "pdf"], dpi=300)
