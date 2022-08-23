import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import os
import string 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup
#import keras 
from tensorflow import keras 
from tensorflow.keras.preprocessing import text,sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM
from wordcloud import WordCloud,STOPWORDS
import warnings
from nltk.corpus import stopwords
nltk.download("stopwords")  
warnings.filterwarnings('ignore')
import nltk 
nltk.download('wordnet')


F_path = os.path.abspath('Fake-subset.csv')
T_path = os.path.abspath('True-subset.csv')

fake_data = pd.read_csv(F_path)
real_data = pd.read_csv(T_path)

#add column 
real_data['target'] = 1
fake_data['target'] = 0 

print(fake_data.head())
print(real_data.head())

#Merging the 2 datasets
data = pd.concat([real_data, fake_data], ignore_index=True, sort=False)
# print(data.head())

data.isnull().sum()

###VISUALISATION
#Count of Fake and Real Data

fig, ax = plt.subplots(1,2, figsize=(10, 4))
g1 = sns.countplot(data.target,ax=ax[0],palette="bright");
g1.set_title("Count of real and fake data")
g1.set_xlabel("Target")
g1.set_ylabel("Count")
g2 = plt.pie(data["target"].value_counts().values,explode=[0,0],labels=data.target.value_counts().index, autopct='%1.1f%%',colors=['Teal','Hotpink'])
fig.show()

#Distribution of The Subject According to Real and Fake Data
plt.figure(figsize=(10, 4))
ax = sns.countplot(x="subject",  hue='target', data=data, palette="bright")
plt.title("Distribution of The Subject of News According to Real and Fake Data")

#### Here Starts the DATA CLEANING ######
data['text']= data['subject'] + " " + data['title'] + " " + data['text']
del data['title']
del data['subject']
del data['date']


# #################################################################
#Removal of HTML Contents
def remove_html(text):
    text = str(text)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removal of Punctuation Marks
def remove_punctuations(text):
    return re.sub('\[[^]]*\]', '', text)

# Removal of Special Characters
def remove_characters(text):
    return re.sub("[^a-zA-Z]"," ",text)

#Removal of stopwords 
#stopwords like is,a,the etc, which do not offer much insight are removed.
#Lemmatization to bring back multiple forms of same word to their common root like 'meeting', 'meets' into 'meet'.
def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word) 
            final_text.append(word)
    return " ".join(final_text)

#Total function
def cleaning(text):
    text = remove_html(text)
    text = remove_punctuations(text)
    text = remove_characters(text)
    text = remove_stopwords_and_lemmatization(text)
    return text

#Apply function on text column
data['text']=data['text'].apply(cleaning)
print(data.head())

# data.to_csv('mergeddata.csv',index =False)


################################################################
#Showing the WordCloud for Real News
plt.figure(figsize = (15,15))
wc = WordCloud(max_words = 500 , width = 1000 , height = 500 , stopwords = STOPWORDS).generate(" ".join(data[data.target == 1].text))
plt.imshow(wc , interpolation = 'bilinear')

#Showing the WordCloud for Fake News
plt.figure(figsize = (15,15))
wc = WordCloud(max_words = 500 , width = 1000 , height = 500 , stopwords = STOPWORDS).generate(" ".join(data[data.target == 0].text))
plt.imshow(wc , interpolation = 'bilinear')

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
text_len=data[data['target']==0]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='Teal')
ax1.set_title('Fake news text')
text_len=data[data['target']==1]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='Hotpink')
ax2.set_title('Real news text')
fig.suptitle('Number of Words in texts')
plt.show()

##Doing N Gram Analysis on the data ( Uni, Bi , Tri ) ##
texts = ' '.join(data['text'])
string = texts.split(" ")
def draw_n_gram(string,i):
    n_gram = (pd.Series(nltk.ngrams(string, i)).value_counts())[:15]
    n_gram_df=pd.DataFrame(n_gram)
    n_gram_df = n_gram_df.reset_index()
    n_gram_df = n_gram_df.rename(columns={"index": "word", 0: "count"})
    print(n_gram_df.head())
    plt.figure(figsize = (10,4))
    return sns.barplot(x='count',y='word', data=n_gram_df)

print(draw_n_gram(string,1))
print(draw_n_gram(string,2))
print(draw_n_gram(string,3))

## Modeling starts from here ##

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], random_state=0)

#Tokenizing Text -> Repsesenting each word by a number #

#keeping all news to 300, added padding to news with less than 300 words and truncating long ones #
max_features = 10000 #Vocabulary Size
maxlen = 300 #Sentence length

###Training LSTM Model
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)
tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
batch_size = 256
epochs = 10
embed_size = 100 #features we want to consider
model = Sequential()
#Non-trainable embeddidng layer
model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False))
#LSTM 
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu')) # Dense used because its a classification problem, Relu used to make sure the gradient doesnot vanish
model.add(Dense(1, activation='sigmoid')) # this will help to classify if it belongs to class 1 or 0
model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, validation_split=0.3, epochs=10, batch_size=batch_size, shuffle=True, verbose = 1)

## Analysis After Training
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")

plt.figure()
plt.plot(history.history["accuracy"], label = "Train")
plt.plot(history.history["val_accuracy"], label = "Test")
plt.title("Accuracy")
plt.ylabel("Acc")
plt.xlabel("epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Loss")
plt.ylabel("Acc")
plt.xlabel("epochs")
plt.legend()
plt.show()

pred = model.predict_classes(X_test)
print(classification_report(y_test, pred, target_names = ['Fake','Real']))