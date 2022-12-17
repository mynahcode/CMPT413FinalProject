import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
import sys
import optparse
import pandas as pd
import csv
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,SpatialDropout1D,SimpleRNN
from keras.layers import Embedding
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from datasets import load_dataset

# Training Dataset Clean up
nltk_stopwords_english = list(stopwords.words('english'))
stopwords_english = list(get_stop_words('en'))
stopwords_english.extend(nltk_stopwords_english)

#Temporary
test_sentiment_labels = [0, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1]

# Architecture
output_dim = 50
dropout = 0.2
num_epochs = 5
batch_size=64
max_len = 32
opt = SGD(learning_rate=0.001) # Set learning_rate to 0.01
classes = 3
train_test_split = 0.7

def loadTrainingData(train_split):
    corpus = []
    dataset = load_dataset("financial_phrasebank", "sentences_allagree") # Labels: 0 -> Negative, 1 -> Neutral, 2 -> Positive
    training_data = dataset['train']
    df = training_data.data.select_columns([0,1]).to_pandas()
    
    i=0
    for sentence in df['sentence']:
        sent = sentence.lower()
        # replace hyphenated words with spaces to dileniate words
        sent = sent.replace("'", "")
        sent = sent.replace("—", "")
        sent = sent.replace("–", "")
        sent = sent.replace("-", "")
        sent = sent.replace("-", "") 
        sent = sent.replace('eur', '')
        sent = sent.replace('mn', '')
        sent = re.sub(r"[^\w\s]", "", sent) # Remove punctuation
        sent = re.sub(r"[0-9]+", "", sent) # Remove digits
        sent = re.sub(r'\s+', ' ', sent) # remove new-line characters
        sent = re.sub(r"\'", "", sent) # remove single quotes
        sent = ' '.join([word for word in sent.split() if len(word) > 1 and sent not in stopwords_english]) #Dont worry about words 'a' or 'i'
        #paragraph_list_noEmpty = [c for c in paragraph_list if c != ' ' and c != '']
        df.loc[i,['sentence']] = [sent]
        i += 1

    i = 0
    for sentence in df['sentence']:
        sentence_word_tokens = word_tokenize(df['sentence'][i])
        corpus += [sentence_word_tokens]
        i += 1
    corpus = sum(corpus, [])
    
    print(f"DEBUG: Corpus Length = {len(corpus)}")
    print(f"DEBUG: Shape of sentiments vector = {df.shape[0]}")

    vocab_size = len(corpus)
    tokenizer = Tokenizer(vocab_size)
    train_size = int(df.shape[0] * train_split)

    #df2['sentiments'] = pd.factorize(df['sentiments'])[0]
     
    x_train = df.sentence[:train_size]
    y_train = df.label[:train_size]
    
    x_test = df.sentence[train_size:]
    y_test = df.label[train_size:]
    
    #print(f"x_train: \n {x_train}\n") 
    #print(f"y_train: \n {y_train}\n")
    #print(f"x_test: \n {x_test} \n")
    #print(f"y_test: \n {y_test} \n")
    
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, max_len, truncating='post', padding='post')
    #print(f"DEBUG: x_train[0] = {x_train[0]}, with length = {len(x_train[0])}")
    
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, max_len, truncating='post', padding='post')
    
    y_train = to_categorical(y_train,3)
    y_test = to_categorical(y_test,3)
    
    #print(f"DEBUG: x_train.shape = {x_train.shape}. y_train.shape = {y_train.shape}")
    #print(f"DEBUG: x_test.shape = {x_test.shape}. y_test.shape = {y_test.shape}")
    
    return x_train, y_train, (x_test, y_test), vocab_size

def SentimentModelLSTM(x_train, y_train, testing_data, vocab_size):
    sentiment_model = Sequential()
    sentiment_model.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length = x_train.shape[1]))
    sentiment_model.add(LSTM(output_dim, dropout=dropout, recurrent_dropout=0.2)) #LSTM w/ 100 memory units
    sentiment_model.add(Dense(classes, activation="softmax"))
    sentiment_model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    sentiment_model.summary()
    
    history = sentiment_model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=testing_data, verbose=2)
    test_history = sentiment_model.evaluate(x_test, y_test)
    
    print(f"Test Score: {test_history[0]}")
    print(f"Test Accuracy: {test_history[1]}")
    
    return sentiment_model
    
def SentimentModelSimpleRNN(x_train, y_train, testing_data, vocab_size):
    sentiment_model = Sequential()
    sentiment_model.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length = x_train.shape[1]))
    sentiment_model.add(SimpleRNN(output_dim, input_length = x_train.shape[1]))
    sentiment_model.add(Dense(3,activation='softmax'))
    
    sentiment_model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    sentiment_model.summary()
    
    history = sentiment_model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=testing_data, verbose=1)
    test_history = sentiment_model.evaluate(x_test, y_test)
    
    print(f"Test Score: {test_history[0]}")
    print(f"Test Accuracy: {test_history[1]}")
    
    return sentiment_model
    
def TestModelsQualitative(testFile, LSTM_model, RNN_model):
    df = pd.read_csv(testFile,index_col=0)
    df = df.dropna() # remove N/A rows in dataframe
    print(df)
    tokenizer = Tokenizer(vocab_size)
    
    for _,line in df.iterrows():
        
        
        input_seq = tokenizer.texts_to_sequences(line['sentences'])
        x_test = pad_sequences(input_seq, max_len)
        #print(f"\n\nTest result for sentiment sentence on LSTM and RNN Model: {line['sentences']}\n")
        y_testLSTM = LSTM_model.predict(x_test, batch_size=1, verbose = 2)[0]
        y_testRNN = RNN_model.predict(x_test, batch_size=1, verbose = 2)[0]
        #result = np.argmax(y_testLSM)
        #result = np.argmax(y_testRNN)
        print(y_testLSTM)
        testLSTM = np.max(y_testLSTM)
        print(np.max(y_testLSTM))
        print(testLSTM == y_testLSTM[0])
        print(testLSTM == y_testLSTM[1])
        print(testLSTM == y_testLSTM[2])
        #Quick implementation
        if(np.argmax(y_testLSTM) == y_testLSTM[0]):
            resultLSTM = "negative"
            if(resultLSTM == line['sentiments']):
                print("correct LSTM")
            else :
                print("incorrect LSTM")
        
        if(np.argmax(y_testLSTM) == y_testLSTM[1]):
            resultLSTM = "neutral"
            if(resultLSTM == line['sentiments']):
                print("correct LSTM")
            else :
                print("incorrect LSTM")
        
        if(np.argmax(y_testLSTM) == y_testLSTM[2]):
            resultLSTM = "positive"
            if(resultLSTM == line['sentiments']):
                print("correct LSTM")
            else :
                print("incorrect LSTM")
        
        if(np.argmax(y_testRNN) == y_testRNN[0]):
            resultRNN = "negative"
            if(resultRNN == line['sentiments']):
                print("correct RNN")
            else:
                print("incorrect RNN")
        if(np.argmax(y_testRNN) == y_testRNN[1]):
            resultRNN = "neutral"
            if(resultRNN == line['sentiments']):
                print("correct RNN")
            else:
                print("incorrect RNN")
        if(np.argmax(y_testRNN) == y_testRNN[2]):
            resultRNN = "positive"
            if(resultRNN == line['sentiments']):
                print("correct RNN")
            else:
                print("incorrect RNN")
            
        
        #if(np.argmax(y_testLSM) == 0):
            
            #print("Predicted Sentiment LSTM: Negative")
            #if()
        #elif(np.argmax(y_testLSM) == 1):
            #print("Predicted Sentiment LSTM: Positive")
            #print(f"Predicted Sentiment value: {result}")
        #if(np.argmax(y_testRNN) == 0):
            #print("Predicted Sentiment RNN: Negative")
            
            #print(f"Predicted Sentiment value: {result}")
        #elif(np.argmax(y_testRNN) == 1):
            #print("Predicted Sentiment RNN: Positive")
            #print(f"Predicted Sentiment value: {result}")
        
    
if __name__ == '__main__':

    x_train, y_train, (x_test, y_test), vocab_size = loadTrainingData(train_test_split)
    sentiment_modelLSTM = SentimentModelLSTM(x_train, y_train, (x_test, y_test), vocab_size)
    sentiment_modelRNN = SentimentModelSimpleRNN(x_train, y_train, (x_test, y_test), vocab_size)
    
    TestModelsQualitative("./data/scraped_data_cleaned.csv", sentiment_modelLSTM, sentiment_modelRNN)