import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

df=pd.read_csv('./naver_news_titles_20240124.csv')
print(df.head())
df.info()
X=df['titles']
Y=df['category']

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform((Y))

label = label_encoder.classes_
print(label)
with open('./models/label_encoder.pickle','wb') as f:
    pickle.dump(label_encoder,f)

onehot_y = to_categorical(labeled_y)

temp=[]
okt= Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
    if i%1000==0:
        print(i)



stopwords = pd.read_csv('./stopwords.csv', index_col = 0)
for j in range(len(X)):
    words=[]
    for i in range(len(X[j])):
        if len(X[j][i]) > 1 :
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])
    X[j] = ' '.join(words)

#토큰나이저를 사용해서 숫자로 변환
token = Tokenizer() #생성
token.fit_on_texts(X) #5번까지 문자리스트를 fit
tokened_x = token.texts_to_sequences((X)) #5번까지 문자리스트를 번호로 변환
wordsize = len(token.word_index)+1
print(tokened_x)
print(wordsize)
#저장 나중에 사용하기 위해(기준)
with open('./models/news_token.pickle','wb') as f:
    pickle.dump(token, f)
## 각 입력 리스트의 형태를 맞춰줌(길이)
max = 0
for i in range(len(tokened_x)):
    if max<len(tokened_x[i]):
        max= len(tokened_x[i])
print(max)

x_pad = pad_sequences(tokened_x, max)
print(x_pad)

#트레인 테스트 데이타 분리
X_train, X_test, Y_train, Y_test = train_test_split(x_pad, onehot_y, test_size=0.2)
print(X_train.shape,Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
xy = np.array(xy,dtype=object)
np.save('./news_data_max_{}_wordsize_{}'.format(max,wordsize),xy)





