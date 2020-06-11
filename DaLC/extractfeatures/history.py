import pandas as pd
features = []
for i in range(8):
    file_name = f'audio_features-part{i}.csv'
    features.append(pd.read_csv(file_name))
features[0].head()
mergedFeatures = pd.concat(features)
mergedFeatures.shape
mergedFeatures.head()
mergedFeatures.count()
mergedFeatures.to_csv('audio_features_liris.csv')
head audio_features_liris.csv
%head audio_features_liris.csv
head audio_features_liris.csv
head 'audio_features_liris.csv'
head?
!head 'audio_features_liris.csv'
mergedFeatures.to_csv('audio_features_liris.csv', index=False)
!head 'audio_features_liris.csv'
%hist -f history.py
