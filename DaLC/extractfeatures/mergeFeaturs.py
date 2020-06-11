'''
由于提取LIRIS 9800个视频的特征过于耗时，单线程大致需要4天,90h。
为了充分利用多核，使用了多进程的提取方式，8个进程，每个大致需要15h，所以需要将各个进程的特征重新进行合并。
将8个csv文件合并。
'''

import pandas as pd

def mergeFeatures(asset):
    features = []
    for i in range(8):
        file_name = f'{asset}_features-part{i}.csv'
        features.append(pd.read_csv(file_name))
    mergedFeatures = pd.concat(features)
    print(mergedFeatures.shape)
    print(mergedFeatures.head())
    print(mergedFeatures.count())
    mergedFeatures.to_csv(f'{asset}_features_liris.csv', index=False)
    return mergedFeatures

if __name__ == '__main__':
    df = []
    for asset in ['audio', 'video']:
        df.append(mergeFeatures(asset))
    total = df[0].merge(df[1], on='id')
    total['name'] = total['id'] + '.mp4'
    total.to_csv('liris_features.csv', index=False)
    print(total.shape)