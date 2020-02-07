#naive bayes###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sal_train = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\naive bayes\\SalaryData_Train.csv")
sal_test = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\naive bayes\\SalaryData_Test.csv")

string_columns = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]


from sklearn import preprocessing
num = preprocessing.LabelEncoder()
for i in string_columns:
    sal_train[i] = num.fit_transform(sal_train[i])
    sal_test[i] = num.fit_transform(sal_test[i])
    
colnames = sal_train.columns
len(colnames[0:13])

sal_test.drop_duplicates(keep='first',inplace=True)
sal_train.drop_duplicates(keep='first',inplace=True)
sal_test.head()
sal_test.describe()
sal_test.columns
sal_test.shape
sal_test.isnull().sum()

sal_train.head()
sal_train.describe()
sal_train.columns
sal_train.shape
sal_train.isnull().sum()

plt.hist(sal_train['age']);plt.xlabel('age');plt.ylabel('Salary');plt.title('histogram of age')
plt.hist(sal_train['education']);plt.xlabel('education');plt.ylabel('Salary');plt.title('histogram of education')
plt.hist(sal_train['sex']);plt.xlabel('sex');plt.ylabel('Salary');plt.title('histogram of sex')
plt.hist(sal_train['workclass']);plt.xlabel('workclass');plt.ylabel('Salary');plt.title('histogram of workclass')
plt.hist(sal_train['maritalstatus']);plt.xlabel('maritalstatus');plt.ylabel('Salary');plt.title('histogram of maritalstatus')
plt.hist(sal_train['occupation']);plt.xlabel('occupation');plt.ylabel('Salary');plt.title('histogram of occupation')
plt.hist(sal_train['relationship']);plt.xlabel('relationship');plt.ylabel('Salary');plt.title('histogram of relationship')
plt.hist(sal_train['race']);plt.xlabel('race');plt.ylabel('Salary');plt.title('histogram of race')
plt.hist(sal_train['capitalgain']);plt.xlabel('capitalgain');plt.ylabel('Salary');plt.title('histogram of capitalgain')
plt.hist(sal_train['capitalloss']);plt.xlabel('capitalloss');plt.ylabel('Salary');plt.title('histogram of capitalloss')
plt.hist(sal_train['hoursperweek']);plt.xlabel('hoursperweek');plt.ylabel('Salary');plt.title('histogram of hoursperweek')
plt.hist(sal_train['native']);plt.xlabel('native');plt.ylabel('Salary');plt.title('histogram of native')

plt.hist(sal_test['age']);plt.xlabel('age');plt.ylabel('Salary');plt.title('histogram of age')
plt.hist(sal_test['education']);plt.xlabel('education');plt.ylabel('Salary');plt.title('histogram of education')
plt.hist(sal_test['sex']);plt.xlabel('sex');plt.ylabel('Salary');plt.title('histogram of sex')
plt.hist(sal_test['workclass']);plt.xlabel('workclass');plt.ylabel('Salary');plt.title('histogram of workclass')
plt.hist(sal_test['maritalstatus']);plt.xlabel('maritalstatus');plt.ylabel('Salary');plt.title('histogram of maritalstatus')
plt.hist(sal_test['occupation']);plt.xlabel('occupation');plt.ylabel('Salary');plt.title('histogram of occupation')
plt.hist(sal_test['relationship']);plt.xlabel('relationship');plt.ylabel('Salary');plt.title('histogram of relationship')
plt.hist(sal_test['race']);plt.xlabel('race');plt.ylabel('Salary');plt.title('histogram of race')
plt.hist(sal_test['capitalgain']);plt.xlabel('capitalgain');plt.ylabel('Salary');plt.title('histogram of capitalgain')
plt.hist(sal_test['capitalloss']);plt.xlabel('capitalloss');plt.ylabel('Salary');plt.title('histogram of capitalloss')
plt.hist(sal_test['hoursperweek']);plt.xlabel('hoursperweek');plt.ylabel('Salary');plt.title('histogram of hoursperweek')
plt.hist(sal_test['native']);plt.xlabel('native');plt.ylabel('Salary');plt.title('histogram of native')

sns.pairplot((sal_train),hue='Salary')
sns.pairplot((sal_test),hue='Salary')

#Q-plot
plt.plot(sal_train);plt.legend(list(sal_train.columns))
plt.plot(sal_test);plt.legend(list(sal_test.columns))

from scipy import stats
corr = sal_train.corr()
corr1 = sal_test.corr()

sns.heatmap(corr, annot=True)
sns.heatmap(corr1, annot=True)

trainX = sal_train[colnames[0:13]]
trainY = sal_train[colnames[13]]
testX  = sal_test[colnames[0:13]]
testY  = sal_test[colnames[13]]

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#Dgnb = GaussianNB()
#Dmnb = MultinomialNB()
sgnb = GaussianNB()
smnb = MultinomialNB()

spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print("Accuracy",(10631+1117)/(10631+1117+2583+729))    #78%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10631+1117)/(10631+1117+2583+729))   #78%

spred_gnb = sgnb.fit(testX,testY).predict(trainX)
confusion_matrix(trainY,spred_gnb)
print("accuracy",(21176+2327)/(21176+2327+1477+5181)) #77%

spred_mnb = smnb.fit(testX,testY).predict(trainX)
confusion_matrix(trainY,spred_mnb)
print("accuracy",(21176+2327)/(21176+2327+1477+5181)) #77%
