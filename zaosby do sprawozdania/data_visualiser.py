import matplotlib.pyplot as plt
import numpy as np

# Dane
data = [
    {"pop_size": 30, "generations": 90, "distances": [149.797, 142.698, 155.689, 162.230]},
    {"pop_size": 30, "generations": 225, "distances": [144.333, 265.181, 148.752, 244.013]},
    {"pop_size": 30, "generations": 360, "distances": [142.298, 141.411, 148.544, 211.768]},
    {"pop_size": 60, "generations": 90, "distances": [259.582, 150.157, 151.509, 147.980]},
    {"pop_size": 60, "generations": 225, "distances": [163.117, 140.606, 149.620, 158.695]},
    {"pop_size": 60, "generations": 360, "distances": [157.552, 210.583, 145.296, 201.296]}
]

# Kolory i etykiety środowisk
colors = ['blue', 'green', 'orange', 'red']
labels = ["Środowisko 1", "Środowisko 2", "Środowisko 3", "Środowisko 4"]

# Wykres
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
axes = axes.flatten()

for i, entry in enumerate(data):
    ax = axes[i]
    x = np.arange(len(entry["distances"]))
    bars = ax.bar(x, entry["distances"], color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_title(f"Pop.: {entry['pop_size']}, Gen.: {entry['generations']}")
    ax.set_ylabel("Długość trasy")
    
    # Dodanie wartości na słupkach
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Globalny tytuł i legenda
#fig.suptitle("Porównanie długości tras w różnych środowiskach", fontsize=16)
#fig.legend(labels, loc="upper right", fontsize=12)
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.show()
# Dodanie wartości na słupkach
