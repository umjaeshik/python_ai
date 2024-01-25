import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

df = pd.read_csv('./crawling_data/naver_headline_news_20240125.csv')
print(df.head())
df.info()

X=df['titles']
Y=df['category']

with open('./models/label_encoder.pickle','rb') as f:
    label_encoder = pickle.load(f)

label = label_encoder.classes_

print(label)

okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i],stem=True)

stopwords = pd.read_csv('./stopwords.csv', index_col = 0)
for j in range(len(X)):
    words=[]
    for i in range(len(X[j])):
        if len(X[j][i]) > 1 :
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

with open('./models/news_token.pickle','rb') as f:
    token = pickle.load(f)

#토큰나이저를 사용해서 숫자로 변환

tokened_x = token.texts_to_sequences((X)) #5번까지 문자리스트를 번호로 변환
for i in range(len(tokened_x)):
    if len(tokened_x[i])>21:
        tokened_x[i] = tokened_x[i][:21]
print((tokened_x))

X_pad = pad_sequences(tokened_x, 21)

model = load_model('./models/news_category_classfication_model_0.8508592844009399.h5')
preds = model.predict(X_pad)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predicts.append([most,second])
df['predict'] = predicts
print(df)

df['OX'] = 0
for i in range(len(df)):
    if df.loc[i,'category'] in df.loc[i,'predict'] :
        df.loc[i,'OX']='O'
    else :
        df.loc[i,'OX']='X'
print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))
