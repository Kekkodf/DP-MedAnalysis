from src.utils.tools import createLogger
from src.classifier import Classifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


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
    
    x_features:List[str] = ['Mutation Count','BM Blast (%)','Hemoglobin (g/dL)','Overall Survival (Months)','Platelet (G/L)','Sex',
                  'Absolute Neutrophil Count (G/L)','Monocyte Count (G/L)','PB Blast (%)','TMB (nonsynonymous)','White Blood Cell Count (G/L)',
                  'Age in years']

    X:np.array = np.array(df[x_features])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info('Feature Variables Created Successfully!')

    #Original Classifier
    og_results = []
    for i in range(100):
        clf:object = Classifier()
        #Train
        clf.fit(X_train, y_train)
        logger.info('Model Fitted Successfully!')
        #Test
        y_pred:np.array = clf.predict(X_test)
        logger.info('Predictions Made Successfully!')
        # Compute the accuracy
        accuracy:np.array = np.mean(y_pred == y_test)
        logger.info('Accuracy Computed Successfully!')
        og_results.append(accuracy)

    #DP Classifier
    private_results_e0 = []
    private_results_e1 = []
    private_results_e2 = []
    private_results_e3 = []
    private_results_e4 = []
    private_results_einf = []
    epsilons = [10,1, 0.1, 0.01, 0.001, 0.0001]
    for i in range(100):
        for eps in epsilons:
            #Train
            clf:object = Classifier(privacyStatus=True, eps=eps)
            clf.fit(X_train, y_train)
            logger.info('Model Fitted Successfully!')
            #Test
            y_pred:np.array = clf.predict(X_test)
            logger.info('Predictions Made Successfully!')
            # Compute the accuracy
            accuracy_private:np.array = np.mean(y_pred == y_test)
            logger.info('Accuracy Computed Successfully!')
            if eps == 1:
                private_results_e1.append(accuracy_private)
            elif eps == 0.1:
                private_results_e2.append(accuracy_private)
            elif eps == 0.01:
                private_results_e3.append(accuracy_private)
            elif eps == 0.001:
                private_results_e4.append(accuracy_private)
            elif eps == 10:
                private_results_einf.append(accuracy_private)
            elif eps == 0.0001:
                private_results_e0.append(accuracy_private)

    data:List[list] = [private_results_e0,private_results_e4, private_results_e3, private_results_e2, private_results_e1, private_results_einf]
    #save results
    ##the df should have a column with the accuracy performance, a column with the epsilon value
    df_results:pd.DataFrame = pd.DataFrame(data = {'Accuracy': private_results_e0 + private_results_e4 + private_results_e3 + private_results_e2 + private_results_e1 + private_results_einf,
                                                    'Epsilon': [0.0001]*100 + [0.001]*100 + [0.01]*100 + [0.1]*100 + [1]*100 + [10]*100})
    df_results.to_csv('results/accuracy_results.csv', index=False)
    palette:object = sns.color_palette("viridis", 6)

    #plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, showmeans=False, palette=palette)
    means = [np.mean(private_results_e0),
             np.mean(private_results_e4), 
             np.mean(private_results_e3), 
             np.mean(private_results_e2), 
             np.mean(private_results_e1), 
             np.mean(private_results_einf)]
    positions = [0, 1, 2, 3, 4, 5]
    for pos, mean in zip(positions, means):
        plt.scatter(pos, mean, color='k', s=50, zorder=5)
        plt.text(pos, mean + 0.005, f'{mean:.2f}', fontsize=12, ha='center')
    #plt.scatter(4, means[-1], color='k', s=50, zorder=5)
    #plt.text(4, means[-1] + 0.005, f'{mean:.2f}', fontsize=12, ha='center')
    plt.plot(positions, means, color='k', linestyle=':', linewidth=1, marker='o', markersize=5, zorder=4)
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=[r'$\varepsilon = 0.0001$',
                                              r'$\varepsilon = 0.001$', 
                                              r'$\varepsilon = 0.01$', 
                                              r'$\varepsilon = 0.1$', 
                                              r'$\varepsilon = 1$', 
                                              r'Original ~ $\varepsilon = 10$'], 
                                              fontsize=12)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Privacy Models', fontsize=14)
    #plt.title('Accuracy Comparison')

    # Save and show the plot
    plt.savefig('results/accuracy_comparison.pdf', format='pdf')
    plt.show()

    logger.info('Plotting Done Successfully!')