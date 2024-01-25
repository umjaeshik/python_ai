import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load('news_data_max_27_wordsize_12545.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(12545,300, input_length=27))# 모든 형태소를 의미공간 상에 벡터화해서 학습. 차원이 커질수록 데이터 값의 밀도 작아짐.(데이터가 희소해짐)-차원의 저주
                                                                    # 12545 차원을 300차원으로 줄임.
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu')) #주변 단어들과의 위치관계
model.add(MaxPooling1D(pool_size=1)) #1이라 없어도 되는데 conv랑 같이 붙여서 씀.
model.add(LSTM(128, activation='tanh', return_sequences=True)) #return_sequences=True : 매번 데이터 나오는거 붙여서 하나의 데이터로 만듬. False면 맨 마지막 하나만 출력.
                                                                    #뒤에 LSTM 있을 경우 줘야함. 마지막은 안줘도 됨.
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='train accuracy')
plt.legend()
plt.show()