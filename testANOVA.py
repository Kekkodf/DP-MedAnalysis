import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    df:pd.DataFrame = pd.read_csv('./results/accuracy_results.csv', sep = ',', header = 0)
    mod = ols('Accuracy ~ Epsilon', data = df).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    #save the results
    aov_table.to_csv('./results/aov.csv')
    print(aov_table)
    #perform multiple pairwise comparison (Tukey HSD)
    mc = MultiComparison(df['Accuracy'], df['Epsilon'])
    result = mc.tukeyhsd()
    
    #plot the results
    fig = result.plot_simultaneous(comparison_name=10, xlabel='Accuracy', ylabel='Privacy Models')
    
    #save the plot
    plt.savefig('./results/tukeyhsd.pdf', format='pdf')
    plt.savefig('./results/tukeyhsd.png', format='png')