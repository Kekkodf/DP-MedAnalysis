import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
sns.set_theme(style="whitegrid")
sns.set_context("talk")


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
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
accuracy:List[float] = []

for eps in epsilons:
    # Train
    model = FeedForwardNeuralNetwork(input_size=len(x_features), hidden_size=150, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        # Forward pass
        outputs = model(torch.tensor(X_train + np.random.laplace(0, 1/eps, X_train.shape)).float())
        loss = criterion(outputs, torch.tensor(y_train).long())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Test - classification
    outputs = model(torch.tensor(X_test).float())
    _, predicted = torch.max(outputs.data, 1)
    mse.append((predicted != torch.tensor(y_test)).sum().item() / len(y_test))
    accuracy.append((predicted == torch.tensor(y_test)).sum().item() / len(y_test))

real_mse:List[float] = []
real_accuracy:List[float] = []

for e in epsilons:
    model = FeedForwardNeuralNetwork(input_size=len(x_features), hidden_size=150, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        # Forward pass
        outputs = model(torch.tensor(X_train).float())
        loss = criterion(outputs, torch.tensor(y_train).long())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Test - classification
    outputs = model(torch.tensor(X_test).float())
    _, predicted = torch.max(outputs.data, 1)
    real_mse.append((predicted != torch.tensor(y_test)).sum().item() / len(y_test))
    real_accuracy.append((predicted == torch.tensor(y_test)).sum().item() / len(y_test))
    

plt.figure(figsize=(10, 6))
plt.plot(epsilons, mse, label='Error', color='#d55e00')
plt.plot(epsilons, real_mse, label='Real Error', linestyle='--', color='#f0e442')
plt.xscale('log')
plt.xlabel(r'$\varepsilon$ (Privacy Budget)')
plt.ylabel('Error')
plt.legend()
plt.grid(False)
plt.tight_layout()

plt.savefig('NeuralNetwork.pdf', format='pdf')
plt.savefig('NeuralNetwork.png', format='png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epsilons, accuracy, label='Accuracy', color='#0072b2')
plt.plot(epsilons, real_accuracy, label='Real Accuracy', linestyle='--', color='#cc79a7')
plt.xscale('log')
plt.xlabel(r'$\varepsilon$ (Privacy Budget)')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(False)
plt.tight_layout()

plt.savefig('NeuralNetworkAccuracy.pdf', format='pdf')
plt.savefig('NeuralNetworkAccuracy.png', format='png')
plt.show()



