from src.utils.tools import createLogger
import pandas as pd

if __name__ == '__main__':
    logger = createLogger()
    logger.info('Logger Created Successfully!')
    df = pd.read_csv('data/BernardEtAl.csv', sep = ',')
    logger.info('Data Loaded Successfully!')
    print(df.head())
