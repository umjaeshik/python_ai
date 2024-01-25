import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

df = pd.read_csv('./naver_news_titles_20240125.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y)
print(labeled_y[:3])
label = label_encoder.classes_ #클래스 나타내줌. 원핫인코더 0과 1로 나타내지니까 있어야함. 매번 토큰할 때마다 숫자 달라서 라벨의 클래스를 나타내줌.
print(label)
with open('./models/label_encoder.pickle','wb') as f:
    pickle.dump(label_encoder, f) #pickle은 저장한 형태 그대로 불러와짐.
onehot_y = to_categorical(labeled_y)
print(onehot_y[:3])
print(X[1:5])
okt= Okt() # 형태소 분리

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True) #stem=True : 원형으로 바뀜.
    if i % 1000:
        print(i)
#print(X[:5])
# 접속사 대명사 감탄사나 한 글자는 학습에 도움이 안되고 방해됨. 불용어.

stopwords = pd.read_csv('./stopwords.csv', index_col=0)
for j in range(len(X)): #문장
    words = []
    for i in range(len(X[j])): #문장 내용 접근
        if len(X[j][i]) > 1: #단어가 한글자 보다 길면
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)
#print(X[:5])

token = Tokenizer() #형태소 하나하나 숫자부여
token.fit_on_texts(X) # 모든 형태소에 라벨링
tokened_x = token.texts_to_sequences(X) # 실제 단어들을 숫자로 이루어진 리스트로 바꿔줌.
wordsize = len(token.word_index) + 1 #인덱스가 1부터 만들어짐. +1한 이유는 나중에 0 쓰려고.-fit on text 떄 못봤던 단어들에 0 붙이거나 패딩할 때 0씀.
#print(tokened_x)
print(wordsize)

with open('./models/news_token.pickle','wb') as f:
    pickle.dump(token, f) #앞쪽에 0을 붙여서 제일 긴 문장에 길이를 맞춰줌. 학습할 때 앞쪽 데이터는 학습 잘 안되므로 앞에 0 붙임.

max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i]) # 가장 긴 문장 저장.

print(max)

x_pad = pad_sequences(tokened_x, max)  #앞쪽에 0을 붙여서 제일 긴 문장에 길이를 맞춰줌. 학습할 때 앞쪽 데이터는 학습 잘 안되므로 앞에 0 붙임.
print(x_pad)

X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test #튜플로 묶임.
xy = np.array(xy, dtype=object)
np.save('./news_data_max_{}_wordsize_{}'.format(max, wordsize), xy)

