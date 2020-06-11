import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack
import random
np.random.seed(2018)
random.seed(2018)

USE_COMMENT = False
audiofeatures = ['audio_harmonic', 'audio_engergy', 'audio_centroid', 'audio_contrast_low',
                 'audio_contrast_middle', 'audio_contrast_high', 'audio_zero_crossing_rate', 'audio_slience_rate']
videofeatures = ['optical_flow_mean', 'optical_flow_std', 'video_warmc', 'video_heavyl', 'video_activep', 'video_hards',
                 'video_darkProportion', 'video_lightPropertion', 'video_saturation', 'video_color_energy', 'video_color_std']
labels = ['arousal', 'excitement', 'pleasure', 'contentment',
          'sleepiness', 'depression', 'misery', 'distress']
df_all = pd.read_csv('./input/filled-labels_features.csv')
cv_id = pd.read_csv('./input/cv_id_10.txt')
df_all['cv_id'] = cv_id['cv_id_10']
# print(df_all.isnull().any())
# for videofea in videofeatures:
#     df_all[videofea].fillna(np.mean(df_all[videofea].values),inplace=True)
# df_all.to_csv('../input/filled-labels_features.csv',index=False)
all_text = df_all['danmaku']

# methods = ['LinerRegression', 'SVR', 'KernelRidge', 'Lasso']
methods = ['GDBT']
# classifer = 'SVR'
# classifer = 'KernelRidge'
# classifer = 'Lasso'


# features = audiofeature     o'
# features = videofeatures
# features_string = 'video'
# features = audiofeatures + videofeatures
# features_string = 'video-audio'

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR,LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
import lightgbm as lgb


for feature in videofeatures + audiofeatures:
    scaler = MinMaxScaler()
    df_all[feature] = scaler.fit_transform(
        df_all[feature].values.reshape(-1, 1))

# print(df_all.head())

for classifer in methods:
    for features, features_string in [(videofeatures, 'video'), (audiofeatures, 'audio'), (videofeatures + audiofeatures, 'videoAndAudio')]:
        test_id = 8
        print('classifer: %s, features: %s, testid: %d' %
              (classifer, features_string, test_id))

        scores = []

        train_data = df_all[df_all['cv_id'] != test_id]  # train data
        valid_data = df_all[df_all['cv_id'] == test_id]  # test data

        sub = pd.DataFrame.from_dict({'id': valid_data['id']})

        for class_name in labels:
            train_target = train_data[class_name].values
            valid_target = valid_data[class_name].values
            train_X = train_data[features]
            test_X = valid_data[features]

            result = None
            if classifer == 'LinerRegression':
                clf = LinearRegression()
            elif classifer == 'SVR':
                # clf = SVR()
                # clf = SVR(C=1.0, epsilon=0.3, kernel='linear')
                # http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search
                svr = SVR(kernel='linear', epsilon=0.3)
                clf = GridSearchCV(svr, {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': [
                                   'linear', 'rbf'], 'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 10, 100], "gamma": np.logspace(-2, 2, 5)})
            elif classifer == 'KernelRidge':
                clf = GridSearchCV(KernelRidge(kernel='linear', gamma=0.1), cv=5,
                                    param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                                "gamma": np.logspace(-2, 2, 5)})
            elif classifer == 'Lasso':
                clf = linear_model.Lasso(alpha=0.1)
            elif classifer == 'GDBT':
                pass
            else:
                raise Error('Variable classifer set error')
            if classifer == 'GDBT':
                # create dataset for lightgbm
                lgb_train = lgb.Dataset(train_X, train_target)

                # specify your configurations as a dict
                params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': {'l2', 'auc'},
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': 0
                }
                gbm = lgb.train(params,
                lgb_train,
                num_boost_round=15)
                gbm.save_model('gdbt-model.txt')
                result = gbm.predict(test_X, num_iteration=gbm.best_iteration)
            else:
                clf.fit(train_X, train_target)
                # print(clf)
                result = clf.predict(test_X)
            result = result.clip(0, 2)
            sub[class_name] = result
            clsscore = mean_squared_error(y_true=valid_target, y_pred=result)
            print(class_name + " scored :" + str(clsscore))
            scores.append(clsscore)

        mean_score = np.mean(scores)
        print("avg : %f" % (mean_score))

        sub.to_csv('%s-baseline-%s-%d.csv' %
                (classifer, features_string, test_id), index=False)

def SbRegressionPredict(test):
    samples, features = test.shape
    result = []
    for i in range(samples):
        result.append(random.uniform(0, 2))
    
    return np.array(result)
