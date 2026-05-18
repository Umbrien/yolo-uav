import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

def plot_diploma_results():
    data = {
        'Method': [
            'Method A\n(RGB Only)',
            'Baseline\n(Thermal Only)',
            'Method B\n(Early Fusion)',
            'Method C\n(Late Fusion)'
        ],
        'mAP50': [91.25, 92.75, 94.45, 96.74],
        'Category': ['Single Modal', 'Single Modal', 'Multi-Modal', 'Multi-Modal']
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#27ae60"]

    ax = sns.barplot(
        x='Method',
        y='mAP50',
        data=df,
        palette=colors,
        edgecolor=".2",
        linewidth=1.5
    )

    ax.set_ylim(85, 97)
    ax.set_ylabel('Mean Average Precision (mAP50), %', fontsize=12, fontweight='bold')
    ax.set_xlabel('Метод детекції', fontsize=12, fontweight='bold')
    plt.title('Порівняння точності методів детекції людей (LLVIP Dataset)', fontsize=14, pad=20)

    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.,
            height + 0.2,
            f'{height:.2f}%',
            ha="center",
            fontsize=12,
            fontweight='bold',
            color='black'
        )

    plt.annotate(
        f'+3.2%',
        xy=(2, 94.45),
        xytext=(0, 92),
        arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=.2"),
        fontsize=11,
        fontweight='bold'
    )

    plt.tight_layout()

    output_file = 'results_chart.png'
    plt.savefig(output_file, dpi=300)
    print(f"✅ Графік збережено як {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_diploma_results()
