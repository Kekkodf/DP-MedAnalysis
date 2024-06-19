import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df:pd.DataFrame = pd.read_csv('./results/accuracy_results.csv', sep = ',', header = 0)
    e1 = df[df['Epsilon'] == 1]['Accuracy']
    e2 = df[df['Epsilon'] == 0.1]['Accuracy']
    e3 = df[df['Epsilon'] == 0.01]['Accuracy']
    e4 = df[df['Epsilon'] == 0.001]['Accuracy']
    f, p = stats.f_oneway(e1, e2, e3, e4)
    print(f, p)