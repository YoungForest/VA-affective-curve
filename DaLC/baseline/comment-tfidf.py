import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR,LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.sparse import hstack
import re
import jieba
import string
import random
np.random.seed(2018)
random.seed(2018)

audiofeatures = ['audio_harmonic','audio_engergy','audio_centroid','audio_contrast_low','audio_contrast_middle','audio_contrast_high','audio_zero_crossing_rate','audio_slience_rate']
videofeatures = ['optical_flow_mean','optical_flow_std','video_warmc','video_heavyl','video_activep','video_hards','video_darkProportion','video_lightPropertion','video_saturation','video_color_energy','video_color_std']
labels = ['arousal','excitement','pleasure','contentment','sleepiness','depression','misery','distress']
comment_fea = ['danmaku_len','danmaku_den']
df_all = pd.read_csv('../input/filled-labels_features.csv')
df_all = df_all.merge(pd.read_csv('../input/comment_length_den.csv'),on=['id'])
cv_id = pd.read_csv('../input/cv_id_10.txt')
df_all['cv_id']=cv_id['cv_id_10']


def text_to_wordlist(query):
    query = re.sub(r'23+',r'哈哈哈',query)
    query = re.sub(r'hh+', r'呵呵', query)
    query = re.sub(r'aa+', r'啊啊啊', query)
    query = re.sub(r'[pr]+', r'喜欢', query)
    query = query.replace(r'艹',r'操')
    query = query.replace(r'屮', r'操')
    query = query.replace(r'芔茻', r'操')
    query = query.replace(r'qwq', r'哭')
    query = query.replace(r'QAQ', r'哭')
    query = query.replace('&', r'和')
    query = query.replace('0', r'零')
    query = query.replace('1', r'一')
    query = query.replace('2', r'二')
    query = query.replace('3', r'三')
    query = query.replace('4', r'四')
    query = query.replace('5', r'五')
    query = query.replace('6', r'六')
    query = query.replace('7', r'七')
    query = query.replace('8', r'八')
    query = query.replace('9', r'九')
    #wordList = jieba.cut(query)
    num = 0
    result = ''
    for sentence in query.split(' '):
        wordlist = jieba.cut(sentence)
        for word in wordlist:
            word = word.rstrip()
            word = word.rstrip('"')
            word = re.sub('([{}“”¨«»®´·º½¾¿¡§£₤‘’\'`])'.format(string.punctuation), r' \1 ', word)
            rexp = re.compile('[^\u4e00-\u9fa5A-Z0-9,。！？~?!.]+', re.IGNORECASE)
            word = re.sub(rexp, r' ', word)
            result = result + ' '+ word
        # if word not in stopwords:
        #     if num == 0:
        #         result = word
        #         num = 1
        #     else:
        #         result = result + ' ' + word
    return result


df_all['danmaku_words'] = df_all['danmaku'].map(lambda x:text_to_wordlist(x))

all_text = df_all['danmaku_words']
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    #strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 5),
    max_features=15000)
word_vectorizer.fit(all_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    #strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 6),
    max_features=20000)
char_vectorizer.fit(all_text)

# l = char_vectorizer.get_feature_names()
# for u in l:
#     print(u)
#print(df_all.head())
test_id = 8
train_data = df_all[df_all['cv_id'] != test_id] # train data
valid_data = df_all[df_all['cv_id'] == test_id] # test data

train_text = train_data['danmaku_words']
test_text = valid_data['danmaku_words']

train_char_features = char_vectorizer.transform(train_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
test_char_features = char_vectorizer.transform(test_text)

train_other_features = train_data[comment_fea].values
test_other_features = valid_data[comment_fea].values

train_features = hstack([train_char_features, train_word_features,train_other_features])
test_features = hstack([test_char_features, test_word_features,test_other_features])

# train_features = hstack([train_char_features, train_word_features])
# test_features = hstack([test_char_features, test_word_features])

sub = pd.DataFrame.from_dict({'id': valid_data['id']})
scores = []
for class_name in labels:
    train_target = train_data[class_name].values
    valid_target = valid_data[class_name].values

    reger = LinearRegression()
    reger.fit(train_features, train_target)
    result = reger.predict(test_features)
    result = result.clip(0,2)
    sub[class_name] = result
    clsscore = mean_squared_error(y_true=valid_target,y_pred=result)
    print(class_name + " scored :" + str(clsscore))
    #print(reger.coef_)
    scores.append(clsscore)
print("avg : " + str(np.mean(scores)))
sub.to_csv('../result/lr-tfidf-comment.csv',index=False)






