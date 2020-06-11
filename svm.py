def uniform_rescale(x):
        return (x / 9800) * 2 - 1

ownFeature = True

def svr(valence, rankLabel):
    if valence:
        value = 'valenceValue'
        rank = 'valenceRank'
        features = '/root/yangsen-data/LIRIS-ACCEDE-features/features/ACCEDEfeaturesValence_TAC2015.txt'
        featureNames = ['colorfulness', 'alpha', 'hueCount', 'maxSaliencyCount', 'compositionalBalance', 'depthOfField', 'saliencyDisparity', 'spatialEdgeDistributionArea', 'entropyComplexity', 'nbWhiteFrames', 'nbFades', 'nbSceneCuts', 'asymmetry_env', 'flatness', 'zcr', 'colorStrength', 'colorRawEnergy']
    else:
        value = 'arousalValue'
        rank = 'arousalRank'
        features = '/root/yangsen-data/LIRIS-ACCEDE-features/features/ACCEDEfeaturesArousal_TAC2015.txt'
        featureNames = ['colorfulness', 'minEnergy', 'alpha', 'lightning', 'globalActivity', 'nbWhiteFrames', 'nbSceneCuts', 'cutLength', 'flatness_env', 'wtf_max2stdratio_1', 'wtf_max2stdratio_2', 'wtf_max2stdratio_3', 'wtf_max2stdratio_4', 'wtf_max2stdratio_5', 'wtf_max2stdratio_6', 'wtf_max2stdratio_7', 'wtf_max2stdratio_8', 'wtf_max2stdratio_9', 'wtf_max2stdratio_10', 'wtf_max2stdratio_11', 'wtf_max2stdratio_12', 'medianLightness', 'slope']
    if ownFeature:
        features = '/root/yangsen/LIRIS-ACCEDE/cluster/DaLC/extractfeatures/liris_features.csv'
        featureNames = ['audio_harmonic','audio_engergy','audio_centroid','audio_contrast_low','audio_contrast_middle','audio_contrast_high','audio_zero_crossing_rate','audio_slience_rate','optical_flow_mean','optical_flow_std','video_warmc','video_heavyl','video_activep','video_hards','video_darkProportion','video_lightPropertion','video_saturation','video_color_energy','video_color_std']
        sep = ','
    else:
        sep = '\t'
 
    import pandas as pd
    featuresDf = pd.read_csv(features, index_col='id', sep=sep)
    ranking = pd.read_csv('/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEranking.txt', sep='\t', index_col='id')
    setsPd = pd.read_csv('/root/yangsen-data/LIRIS-ACCEDE-annotations/LIRIS-ACCEDE-annotations/annotations/ACCEDEsets.txt', sep='\t', index_col='id')
    withSetSplitLabel = pd.merge(pd.merge(ranking, setsPd, on='name'), featuresDf, on='name')
    withSetSplitLabel['uniform'] = uniform_rescale(withSetSplitLabel[rank])
    trainSet = withSetSplitLabel[withSetSplitLabel['set'] >= 1]
    testSet = withSetSplitLabel[withSetSplitLabel['set'] == 0]
    from sklearn.svm import SVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    if rankLabel:
        value = 'uniform'
    y = trainSet[value]
    X = trainSet[featureNames]
    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(X, y)
    Ytest = testSet[value]
    Xtest = testSet[featureNames]
    Ypredict = regr.predict(Xtest)
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(Ytest, Ypredict)


if __name__=='__main__':
    print (f'valence: {svr(True, False)}')
    print (f'arousal: {svr(False, False)}')
    print (f'valence: {svr(True, True)}')
    print (f'arousal: {svr(False, True)}')
