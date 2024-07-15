import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.cluster import KMeans
sns.set_theme(style="whitegrid")
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set_palette("husl")
# Load the data
df:pd.DataFrame = pd.read_csv('./data/BernardEtAl.csv', sep=',')
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
x_features:List[str] = ['Mutation Count','BM Blast (%)','Hemoglobin (g/dL)','Overall Survival (Months)','Platelet (G/L)','Sex',
                      'Absolute Neutrophil Count (G/L)','Monocyte Count (G/L)','PB Blast (%)','White Blood Cell Count (G/L)',
                      'Age in years']
x_labels:List[str] = ['Mutation Count','BM Blast','Hemoglobin','Survival (Months)','Platelet','Sex',
                        'ANC','Monocyte Count','PB Blast','WBC',
                        'Age']
X = df[x_features]
#map df columns names to x_labels
X.columns = x_labels
cor = X.corr()
plt.figure(figsize=(12, 10))
ax = sns.heatmap(cor, annot=False, fmt=".2f", cbar_kws={'label': 'Correlation Coefficient'}, cmap='coolwarm')
cbar = ax.collections[0].colorbar
cbar.set_label('Correlation Coefficient', size=18)
cbar.ax.tick_params(labelsize=18)
plt.xticks(size=21)
plt.yticks(rotation=0, size=21)
plt.tight_layout()
plt.savefig('./correration_matrix.png')
plt.show()

# Standardize the data
def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(X, 2)
kmeans = pd.DataFrame(clust_labels)
X.insert((X.shape[1]), 'kmeans', kmeans)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(X['ANC'], X['WBC'], c=kmeans[0], s=50, cmap='viridis')
#ax.cbar = plt.colorbar(scatter)
ax.set_title('K-Means Clustering')
ax.set_xlabel('ANC')
ax.set_ylabel('WBC')
plt.tight_layout()
plt.grid(False)
plt.savefig('./kmeans.png')
plt.show()
