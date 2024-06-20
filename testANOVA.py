import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df:pd.DataFrame = pd.read_csv('./results/accuracy_results.csv', sep = ',', header = 0)
    mod = ols('Accuracy ~ Epsilon', data = df).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table.to_latex())