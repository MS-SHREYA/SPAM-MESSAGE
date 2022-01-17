from imaplib import Int2AP
import numpy as np
import pandas as pd 
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
path = 'C:/Users/msshr/OneDrive/Pictures/Screenshots/campK12/Spam_Message_Classifier-main/spam.csv'
df_sms = pd.read_csv(path, encoding = 'latin-1')
df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
df_sms = df_sms.rename(columns = {'v1' : 'label', 'v2': 'sms'})
df_sms['length'] = df_sms['sms'].apply(len)
df_sms.loc[:, 'label'] = df_sms.label.map({'ham': 0, 'spam' : 1})

X_train, X_test, y_train, y_test = train_test_split(
    df_sms['sms'], df_sms['label'], test_size = 0.20, random_state = 1
) #split into test and train data to train itself
count_vector = CountVectorizer()     #count vectorizer converts text into matrix
training_data = count_vector.fit_transform(X_train)  
testing_data = count_vector.transform(X_test) 

naive_bayes = MultinomialNB()    #multinomialNB classifies words  
naive_bayes.fit(training_data, y_train)
MultinomialNB(alpha = 1.0, class_prior = None, fit_prior = True)  # prior probabiliity before collecting new evidence
predictions = naive_bayes.predict(testing_data)
user_input = input("Enter a  message: ")
inp = np.array(user_input)
inp = np.reshape(inp, (1,-1))
inp_conv = count_vector.transform(inp.ravel()) 
result = naive_bayes.predict(inp_conv)

for element in result :
    if result[0] == 0:
        print("It's not a spam")
    else :
        print("It's a spam")