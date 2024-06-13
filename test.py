from src.utils.tools import createLogger
from src.classifier import Classifier
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    logger:object = createLogger()
    logger.info('Logger Created Successfully!')
    df:pd.DataFrame = pd.read_csv('data/BernardEtAl.csv', sep = ',', header = 0)
    logger.info('Data Loaded Successfully!')  
    #preoprocessing
    #remove the rows with missing values
    df['Overall Survival Status'] = df['Overall Survival Status'].map({'0:LIVING': 0, '1:DECEASED': 1})
    y:np.array = np.array(df['Overall Survival Status'])
    logger.info('Target Variable Created Successfully!')

    df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
    
    x_features = ['Mutation Count','BM Blast (%)','Hemoglobin (g/dL)','Overall Survival (Months)','Platelet (G/L)','Sex',
                  'Absolute Neutrophil Count (G/L)','Monocyte Count (G/L)','PB Blast (%)','TMB (nonsynonymous)','White Blood Cell Count (G/L)',
                  'Age in years']

    X = np.array(df[x_features])
    logger.info('Feature Variables Created Successfully!')
    
    clf = Classifier()
    clf.fit(X, y)
    #print(clf.predict(X))
    original = np.array(clf.predict(X))

    clf = Classifier(privacyStatus='True', eps=0.2)
    clf.fit(X, y)
    #print(clf.predict(X))
    private = np.array(clf.predict(X))

    #plot the results in histograms grouping the labels on the x axis by ['original', 'private']
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(original, ax=ax[0], kde=False)
    ax[0].set_title('Original')
    sns.histplot(private, ax=ax[1], kde=False)
    ax[1].set_title('Private')
    ax[0].set_xlabel('Labels Predicted')
    ax[1].set_xlabel('Labels Predicted')
    ax[0].set_ylabel('Count')
    ax[1].set_ylabel('Count')
    ax[0].set_xticks([0, 1])
    ax[1].set_xticks([0, 1])
    ax[0].set_xticklabels(['Living', 'Deceased'])
    ax[1].set_xticklabels(['Living', 'Deceased'])
    plt.show()


