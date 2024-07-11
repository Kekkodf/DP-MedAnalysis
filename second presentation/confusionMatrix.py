import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
sns.set_theme(style="whitegrid")
sns.set_context("talk")

import torch
import torch.nn as nn

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

#for each epsilon train the model and calculate the confusion matrix on the test set

for eps in epsilons:
    break
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
    #calculate the confusion matrix
    cm = confusion_matrix(y_test, predicted)
    #normalize the confusion matrix between 0 and 1 with 3 decimal points
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    cm = [[round(j, 3) for j in i] for i in cm]
    #plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, vmax=1, cbar_kws={'format': '%.1f'})
    plt.xlabel('Predicted labels')
    plt.xticks([0.5, 1.5], ['Living', 'Deceased'])
    plt.yticks([0.5, 1.5], ['Living', 'Deceased'])
    plt.ylabel('True labels')
    plt.title(r'$\varepsilon$ (Privacy Budget) = ' + str(eps))
    plt.tight_layout()
    plt.savefig(f'./second presentation/confusion_matrix_{eps}.png', format='png')
    plt.savefig(f'./second presentation/confusion_matrix_{eps}.pdf', format='pdf')
    plt.show()
    #calculate the accuracy
    accuracy = np.round(np.trace(cm) / float(np.sum(cm)), 3)
    print('Accuracy: ', accuracy)
    print('Misclassification Rate: ', 1 - accuracy)
    print('---------------------------------')

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
#calculate the confusion matrix
cm = confusion_matrix(y_test, predicted)
#normalize the confusion matrix between 0 and 1 with 3 decimal points
cm = cm / cm.sum(axis=1)[:, np.newaxis]
cm = [[round(j, 3) for j in i] for i in cm]
#plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, vmax=1, cbar_kws={'format': '%.1f'})
plt.xlabel('Predicted labels')
plt.xticks([0.5, 1.5], ['Living', 'Deceased'])
plt.yticks([0.5, 1.5], ['Living', 'Deceased'])
plt.ylabel('True labels')
plt.title('Real Scenario')
plt.tight_layout()
plt.savefig(f'./second presentation/confusion_matrix.png', format='png')
plt.savefig(f'./second presentation/confusion_matrix.pdf', format='pdf')
plt.show()
#calculate the accuracy
accuracy = np.round(np.trace(cm) / float(np.sum(cm)), 3)
print('Accuracy: ', accuracy)
print('Misclassification Rate: ', 1 - accuracy)
print('---------------------------------')