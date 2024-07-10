import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.set_context("talk")

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
df:pd.DataFrame = pd.read_csv('./data/BernardEtAl.csv', sep=',')
df['Overall Survival Status'] = df['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})
y:np.array = np.array(df['Overall Survival Status'])
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
x_features:List[str] = ['Mutation Count','BM Blast (%)','Hemoglobin (g/dL)','Overall Survival (Months)','Platelet (G/L)','Sex',
                      'Absolute Neutrophil Count (G/L)','Monocyte Count (G/L)','PB Blast (%)','TMB (nonsynonymous)','White Blood Cell Count (G/L)',
                      'Age in years']

X:np.array = np.array(df[x_features])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

epsilons:List[int] = [0.0001, 0.001, 0.01, 0.1, 1, 10]
mse:List[float] = []
mae:List[float] = []
for eps in epsilons:
    # Train
    clf:object = LinearRegression()
    clf.fit(X_train + np.random.laplace(0, 1/eps, X_train.shape), y_train)
    # Test
    y_pred:np.array = clf.predict(X_test)
    mse.append(mean_squared_error(y_test, y_pred))
    mae.append(mean_absolute_error(y_test, y_pred))

real_mse:List[float] = []
real_mae:List[float] = []

for e in epsilons:
    clf:object = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred:np.array = clf.predict(X_test)
    real_mse.append(mean_squared_error(y_test, y_pred))
    real_mae.append(mean_absolute_error(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.plot(epsilons, mse, label='Mean Squared Error', color='#d55e00')
plt.plot(epsilons, mae, label='Mean Absolute Error', color='#0072b2')
plt.plot(epsilons, real_mse, label='Real Mean Squared Error', linestyle='--', color='#f0e442')
plt.plot(epsilons, real_mae, label='Real Mean Absolute Error', linestyle='--', color='#cc79a7')
plt.xscale('log')
plt.xlabel(r'$\varepsilon$ (Privacy Budget)')
plt.ylabel('Error')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('LinearRegression.pdf', format='pdf')
plt.show()


    
