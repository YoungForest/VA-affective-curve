import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import pickle as pickle
import re
import jieba
import string
# import scipy
LABELFILE = '../input/filled-labels_features.csv'

def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)
def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)
    return matrix

def text_to_wordlist(query):
    query = re.sub(r'23+',r'哈哈哈',query)
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
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')
        word = re.sub('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation), r' \1 ', word)
        rexp = re.compile('[^\u4e00-\u9fa5A-Z0-9,。！？~?!.]+',re.IGNORECASE)
        word = re.sub(rexp, r' ', word)
        result = result + ' ' + word
        # if word not in stopwords:
        #     if num == 0:
        #         result = word
        #         num = 1
        #     else:
        #         result = result + ' ' + word
    return result

def getDanmaku_den(query):
    query = re.sub(r'23+', r'哈哈哈', query)
    query = re.sub(r'[pr]+', r'喜欢', query)
    query = query.replace(r'艹', r'操')
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
    damaku_list = query.split(' ')
    return len(damaku_list)

# txt = '前六百qwq 前50？ 233333333 开口苏！ 好好听 好棒 王朝大大好棒 好听 哦哦哦哦哦 男神唱这首啊 棒 我是来表白的 王朝大大！！！ 卧槽 开口跪 好听233 好听 好听哭QAQ 好听！！ 好听 棒 好好听 开口跪美哭啊啊啊啊啊啊啊啊啊 啊啊啊啊'
#
# txt = re.sub(r'23+',r'哈哈哈',txt)
# print(txt)
#print(text_to_wordlist(txt))
df = pd.read_csv(LABELFILE)
print(len(df))

df['danmaku'] = df['danmaku'].map(lambda x:text_to_wordlist(x))
df['danmaku_len']=df['danmaku'].map(lambda x : len(x))
df['danmaku_den']=df['danmaku'].map(lambda x:getDanmaku_den(x))
print(df.head())
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    #strip_accents='unicode',
    analyzer='word',
    #token_pattern=r'\w{1,}',
    ngram_range=(1, 5),
    max_features=15000)

all_text = df['danmaku']

word_vectorizer.fit(all_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    #strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 6),
    max_features=20000)
char_vectorizer.fit(all_text)




l = word_vectorizer.get_feature_names()
# for u in l:
#     print(u)
word_features = word_vectorizer.transform(all_text)
char_features = char_vectorizer.transform(all_text)

danmaku_tfidf_features = hstack([word_features,char_features])
save_pickle(danmaku_tfidf_features,'tfidf_features.pkl')

print(word_features.shape)
print(char_features.shape)

ids = df['id'].values
danmaku_lens = df['danmaku_len'].values
danmaku_den =df['danmaku_den'].values
#newdf = pd.DataFrame(columns=['ID','danmaku_len','word_feature','char_feature'])
newdf = pd.DataFrame(columns=['id','danmaku_len'])

newdf['id']=ids
newdf['danmaku_len']= danmaku_lens
newdf['danmaku_den']=danmaku_den
print(len(newdf))
newdf.to_csv('comment_length_den.csv',index=False)
# newdf.to_pickle('comment_features.pkl')

