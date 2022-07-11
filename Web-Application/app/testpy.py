import pandas as pd
import numpy as np

# text preprocessing
from nltk.tokenize import word_tokenize
import re

# preparing input to our model
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

# keras layers
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense


# Number of labels: joy, anger, fear, sadness, neutral
num_classes = 5

# Number of dimensions for word embedding
embed_num_dims = 300

# Max input length (max number of words) 
max_seq_len = 500

class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

# import os
# os.chdir('data')


data_train = pd.read_csv('C:/Users/Dell/Desktop/Emotion_detection/data/data_train.csv', encoding='utf-8')
data_test = pd.read_csv('C:/Users/Dell/Desktop/Emotion_detection/data/data_test.csv', encoding='utf-8')

X_train = data_train.Text
X_test = data_test.Text

y_train = data_train.Emotion
y_test = data_test.Emotion

data = data_train.append(data_test, ignore_index=True)


def clean_text(data):
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    # tekenization using nltk
    data = word_tokenize(data)
    
    return data





import nltk
# nltk.download('punkt')
texts = [' '.join(clean_text(text)) for text in data.Text]

texts_train = [' '.join(clean_text(text)) for text in X_train]
texts_test = [' '.join(clean_text(text)) for text in X_test]



tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequence_train = tokenizer.texts_to_sequences(texts_train)
sequence_test = tokenizer.texts_to_sequences(texts_test)

index_of_words = tokenizer.word_index

# vacab size is number of unique words + reserved 0 index for padding
vocab_size = len(index_of_words) + 1

# print('Number of unique words: {}'.format(len(index_of_words)))


X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )

# X_train_pad


# import os
# os.chdir('data')


from keras.models import load_model
new_model = load_model('C:/Users/Dell/Desktop/Emotion_detection/model/trail_model.h5')


new_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# new_model.summary()
# new_model.get_weights()


# import time
import re
# sentence = 'I really like python. it is worst day'
# sentence = 'Emotion plays a major role in influencing our behaviour. Life would be dreary without feelings like joy and sorrow, excitement and disappointment, love and fear, hope and dismay. Emotion adds colour and spice to life.'
def prediction_by_list(sentence):
# sentence = 'I really like python. it is worst day'
  sen = re.split("(?<!\d)[,.?!](?!\d)",sentence)
  pred_result=[]
  print(sen)
  result_message=[]
  for i in sen:
    messag = i
    result_message.append(messag)
      # print(messag)
    message=[]
    message.append(i)
      
      # print(message)
      # lis = 
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)

    pred = new_model.predict(padded)

    a=( str(message))
    b=((class_names[np.argmax(pred)]))
    pred_result.append(b)
    

  def countList(result_message, pred_result):
    return [sub[item] for item in range(len(pred_result))
                      for sub in [result_message, pred_result]]
  
  return(countList(result_message, pred_result))

  # sentence = 'I really like python. it is worst day'
sentence = 'Emotion plays a major role in influencing our behaviour. Life would be dreary without feelings like joy and sorrow, excitement and disappointment, love and fear, hope and dismay. Emotion adds colour and spice to life.'
def prediction_overall(sentence):
    message=[]
    message.append(sentence)
    
    seq = tokenizer.texts_to_sequences(message)
    padded = pad_sequences(seq, maxlen=max_seq_len)

    pred = new_model.predict(padded)
    b=((class_names[np.argmax(pred)]))
    
    return b

  
prediction_overall(sentence)

