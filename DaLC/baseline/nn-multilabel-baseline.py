import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.sparse import hstack
import keras
from keras.layers import Input,Dense,Dropout
from keras.optimizers import Adam
from keras.models import Model
#from keras.activations import relu,linear,sigmoid
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
import random
np.random.seed(2018)
random.seed(2018)

USE_COMMENT = False
audiofeatures = ['audio_harmonic','audio_engergy','audio_centroid','audio_contrast_low','audio_contrast_middle','audio_contrast_high','audio_zero_crossing_rate','audio_slience_rate']
videofeatures = ['optical_flow_mean','optical_flow_std','video_warmc','video_heavyl','video_activep','video_hards','video_darkProportion','video_lightPropertion','video_saturation','video_color_energy','video_color_std']
labels = ['arousal','excitement','pleasure','contentment','sleepiness','depression','misery','distress']
comment_fea = ['danmaku_len']
df_all = pd.read_csv('../input/filled-labels_features.csv')
df_all = df_all.merge(pd.read_csv('../input/comment_features.csv'),on=['id'])
cv_id = pd.read_csv('../input/cv_id_10.txt')
df_all['cv_id']=cv_id['cv_id_10']
#print(df_all.isnull().any())
# for videofea in videofeatures:
#     df_all[videofea].fillna(np.mean(df_all[videofea].values),inplace=True)
# df_all.to_csv('../input/filled-labels_features.csv',index=False)
all_text = df_all['danmaku']

features = videofeatures + audiofeatures
#features = audiofeatures #+ comment_fea
feanum = len(features)

scores = []
def buildmodel():
    input_fea = Input(shape=(feanum,))
    x = Dense(30, activation='sigmoid')(input_fea)
    x3 = Dense(100,activation='relu')(input_fea)
    x1 = concatenate([x, x3])
    x1 = Dropout(0.1)(x1)
    x = Dense(30, activation='sigmoid')(x1)
    x2 = Dense(11, activation='linear')(x1)
    x3 = Dense(11,activation='relu')(x1)
    x1 = concatenate([x, x2, x3])
    x = Dropout(0.1)(x1)
    preds = Dense(8,activation='linear')(x)
    model = Model([input_fea], preds)
    adam = Adam(lr=2e-3)
    # model.compile(loss=root_mean_squared_error, optimizer=adam, metrics=['mae'])
    model.compile(loss='mse', optimizer=adam, metrics=['mse'])
    return model

from sklearn.preprocessing import MinMaxScaler
for feature in features:
    scaler = MinMaxScaler()
    df_all[feature] = scaler.fit_transform(df_all[feature].values.reshape(-1,1))
#print(df_all.head())
test_id = 8
train_data = df_all[df_all['cv_id'] != test_id] # train data
valid_data = df_all[df_all['cv_id'] == test_id] # test data

sub = pd.DataFrame.from_dict({'id': valid_data['id']})


train_target = train_data[labels].values
valid_target = valid_data[labels].values
train_X = train_data[features]
test_X = valid_data[features]
model = buildmodel()
bst_model_path = '../nnmodel/multilabelnn.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit(train_X, train_target,
                 validation_data=(test_X, valid_target),
                 epochs=50, batch_size=32, shuffle=True,
                 callbacks=[early_stopping, model_checkpoint],verbose=False)

model.load_weights(bst_model_path)

result = model.predict(test_X)
result = result.clip(0,2)
print(result.shape)
for i in range(len(labels)):
    class_name = labels[i]
    sub[class_name] = result[:,i]
    y_t = valid_target[:,i]
    preds = result[:,i]
    clsscore = mean_squared_error(y_true=y_t,y_pred=preds)
    print(class_name + " scored :" + str(clsscore))
    #print(reger.coef_)
    scores.append(clsscore)
print("avg : " + str(np.mean(scores)))
sub.to_csv('nn-baseline-multilabel-av.csv',index=False)






