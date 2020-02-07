import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

sms = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/naive bayes/sms_raw_NB.csv",encoding = "ISO-8859-1")
sms.head()
sms.shape
sms.info()
sms.isnull().sum() 
sms.drop_duplicates(keep='first', inplace=True) 
sms.dtypes
sms.type.value_counts()
sms.columns

import seaborn as sns
sns.countplot(sms['type']).set_title('Count of ham & spam')
plt.hist(sms.type)


#help(sns.boxplot)
import re
stop_words=[]
with open("C:/Users/ADMIN/Desktop/Data_Science_Assig/stop.txt") as f:
    stop_words = f.read()

# splitting the entire string by giving separator as "\n" to get list of 
# all stop words
stop_words = stop_words.split("\n")


"this is awsome 1231312 $#%$# a i he yu nwj"

def cleaning_text(i):
   i=re.sub("[0-9" "]+"," ",i)
   i=re.sub("[^A-Za-z" "]+"," ",i).lower()
   w = []
   for word in i.split(" "):
        if len(word)>3:
            w.append(word)
   return (" ".join(w))


"This is Awsome 1231312 $#%$# a i he yu nwj".split(" ")

cleaning_text("This is Awsome 1231312 $#%$# a i he yu nwj")


cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .. are you?")

sms.text = sms.text.apply(cleaning_text)

# removing empty rows 
sms.shape
sms = sms.loc[sms.text != " ",:]

def split_into_words(i):
    return [word for word in i.split(" ")]

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
sms_train,sms_test = train_test_split(sms,test_size=0.3)

# Preparing sms texts into word count matrix format 
sms_bow = CountVectorizer(analyzer = split_into_words)
sms_bow.fit(sms.text)


# For all messages
all_sms_matrix = sms_bow.transform(sms.text)
all_sms_matrix.shape
# For training messages
train_sms_matrix = sms_bow.transform(sms_train.text)
train_sms_matrix.shape    

# For testing messages
test_sms_matrix = sms_bow.transform(sms_test.text)
test_sms_matrix.shape      


####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_sms_matrix,sms_train.type)
train_pred_m = classifier_mb.predict(train_sms_matrix)
accuracy_train_m = np.mean(train_pred_m==sms_train.type) 
accuracy_train_m

test_pred_m = classifier_mb.predict(test_sms_matrix)
accuracy_test_m = np.mean(test_pred_m==sms_test.type) 
accuracy_test_m
# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_sms_matrix.toarray(),sms_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_sms_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==sms_train.type) 
accuracy_train_g

test_pred_g = classifier_gb.predict(test_sms_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==sms_test.type) 
accuracy_test_g
#########################################################3
# Learning Term weighting and normalizing on entire sms
tfidf_transformer = TfidfTransformer().fit(all_sms_matrix)

# Preparing TFIDF for train sms
train_tfidf = tfidf_transformer.transform(train_sms_matrix)

train_tfidf.shape # (3891, 6661)

# Preparing TFIDF for test sms
test_tfidf = tfidf_transformer.transform(test_sms_matrix)

test_tfidf.shape #  (1668, 6661)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,sms_train.type)
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m==sms_train.type) 
accuracy_train_m

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==sms_test.type) 
accuracy_test_m
# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),sms_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g = np.mean(train_pred_g==sms_train.type) 
accuracy_train_g
test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g==sms_test.type) 
accuracy_test_g