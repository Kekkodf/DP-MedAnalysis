import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
#import to cluster using DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
sns.set_theme(style="whitegrid")
sns.set_context("talk")

# Load the data
df:pd.DataFrame = pd.read_csv('./data/BernardEtAl.csv', sep=',')
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
x_features:List[str] = ['BM Blast (%)','Hemoglobin (g/dL)']

X:np.array = np.array(df[x_features])




