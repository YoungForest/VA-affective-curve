import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error

def uniform_rescale(x):
    return (x / 9800) * 2 - 1

def uniformRandomModel(valence, rankLabel):
    if valence:
        value = 'valenceValue'
        rank = 'valenceRank'
    else:
        value = 'arousalValue'
        rank = 'arousalRank'
    if rankLabel:
        value = 'uniform'
    rankingPd = pd.read_csv('/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt', sep='\t', index_col='id')
    setsPd = pd.read_csv('/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEsets.txt', sep='\t', index_col='id')
    withSetSplitLabel = pd.merge(rankingPd, setsPd, on='name')
    withSetSplitLabel['uniform'] = uniform_rescale(withSetSplitLabel[rank])
    trainSet = withSetSplitLabel[withSetSplitLabel['set'] >= 1]
    testSet = withSetSplitLabel[withSetSplitLabel['set'] == 0]
    y = trainSet[value]
    Ytest = testSet[value]
    begin = time.time()
    if rank:
        Ypredict = np.random.uniform(-1.0, 1.0, Ytest.shape)
    else:
        Ypredict = np.random.uniform(0.0, 5.0, Ytest.shape)
    end = time.time()
    cost = end - begin
    print(
        f'total: {cost}, length: {len(Ypredict)}, cost per clip: {cost / len(Ypredict)}')
    return mean_squared_error(Ytest, Ypredict)



if __name__ == '__main__':
    print(f'valence: {uniformRandomModel(True, False)}')
    print(f'arousal: {uniformRandomModel(False, False)}')
    print(f'valence: {uniformRandomModel(True, True)}')
    print(f'arousal: {uniformRandomModel(False, True)}')