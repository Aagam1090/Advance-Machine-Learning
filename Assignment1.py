import nltk
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


path = 'bbc'
df = pd.DataFrame(columns=['Category','text'])
for r,d,f in os.walk(path):
    for file in f:
        p = os.path.join(r, file)
        text = open(p,"r+") 
        cat = os.path.basename(os.path.dirname(p))
        txt = text.read()
        df.loc[-1] = [cat,txt]  # adding a row
        df.index = df.index + 1  # shifting index
        df = df.sort_index()  
        
le=preprocessing.LabelEncoder()
df['Category']= le.fit_transform(df['Category']) 


df.to_csv(r"TextClassification.csv")

df = pd.read_csv('TextClassification.csv')

def lemmatize(txt):
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer() 
    return lemmatizer.lemmatize(txt)
#label encoding categorical variable- Category
    
le = LabelEncoder() 

X_train, X_test, y_train, y_test = train_test_split(df['text'],df['Category'], random_state=1)


cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

clf = RandomForestClassifier(n_estimators=100).fit(X_train_cv,y_train)
print(clf.score(X_test_cv,y_test))
from sklearn.metrics import accuracy_score, precision_score, recall_score
print('Accuracy score: ', accuracy_score(y_test, predictions))


