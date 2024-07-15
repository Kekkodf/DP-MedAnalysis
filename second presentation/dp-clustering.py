import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set_theme(style="whitegrid")
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load the data
df: pd.DataFrame = pd.read_csv('./data/BernardEtAl.csv', sep=',')
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})

x_features: List[str] = [
    'Mutation Count', 'BM Blast (%)', 'Hemoglobin (g/dL)', 'Overall Survival (Months)',
    'Platelet (G/L)', 'Sex', 'Absolute Neutrophil Count (G/L)', 'Monocyte Count (G/L)',
    'PB Blast (%)', 'White Blood Cell Count (G/L)', 'Age in years'
]
x_labels: List[str] = [
    'Mutation Count', 'BM Blast', 'Hemoglobin', 'Survival (Months)', 'Platelet', 'Sex',
    'ANC', 'Monocyte Count', 'PB Blast', 'WBC', 'Age'
]

X = df[x_features]

# Create a span of epsilons from 0.0001 to 10, with -1 for no noise (Real data)
epsilons = [-1, 10, 1, 0.1, 0.01, 0.001, 0.0001]
k = [2, 3, 4, 5, 6, 7, 8, 9, 10]

silhouette_scores = []

for eps in epsilons:
    for i in k:
        if eps == -1:
            noise = np.zeros(X.shape)
        else:
            noise = np.random.laplace(0, 1/eps, X.shape)
        X_noisy = X + noise
        kmeans = KMeans(n_clusters=i, random_state=0).fit(X_noisy)
        labels = kmeans.labels_
        score = silhouette_score(X_noisy, labels)
        silhouette_scores.append(score)

silhouette_scores = np.array(silhouette_scores).reshape(len(epsilons), len(k))

fig, ax = plt.subplots(figsize=(12, 10))
# Plot a 2D heatmap where x is the number of clusters, y is the epsilon, and z is the silhouette score
sns.heatmap(silhouette_scores, ax=ax, xticklabels=k, yticklabels=epsilons[::-1], annot=True, fmt=".2f", cmap='coolwarm')
cbar = ax.collections[0].colorbar
cbar.set_label('Silhouette Score', size=18)

# Replace the y-tick label for eps == -1 with 'Real'
y_ticks = ['Real' if eps == -1 else str(eps) for eps in epsilons]
ax.set_yticklabels(y_ticks)
plt.yticks(rotation=0, size=18)
plt.xticks(size=18)

plt.xlabel('Number of Clusters', size=21)
plt.ylabel(r'$\varepsilon$ (Privacy Budget)', size=21)
#plt.title('Silhouette Scores in Different Privacy Scenarios')
plt.tight_layout()
plt.savefig('./second presentation/silhouette_scores.png')
plt.savefig('./second presentation/silhouette_scores.pdf')
plt.show()
