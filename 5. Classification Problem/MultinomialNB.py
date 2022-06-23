import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

print("-------------------------- Multinomial --------------------------")
sms = pd.read_csv("spam.csv",delimiter=",",encoding = "ISO-8859-1")
print(sms.keys())
sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)
print(sms.head(5))

sms['label_num'] = sms.v1.map({'ham':0, 'spam':1})
print(sms.head())

X = sms.v2
y = sms.label_num


X_train = X[0:4179]
X_test = X[4179:]
y_train = y[0:4179]
y_test = y[4179:]

vect = CountVectorizer()
vect.fit(X_train)

X_train_dtm = vect.transform(X_train)
X_test_dtm = vect.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)

print("Accuracy : ", metrics.accuracy_score(y_test, y_pred_class))
print("Confusion Matrix : \n", metrics.confusion_matrix(y_test, y_pred_class))

"""

-------------------------- Multinomial --------------------------
Index(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], dtype='object')
     v1                                                 v2
0   ham  Go until jurong point, crazy.. Available only ...
1   ham                      Ok lar... Joking wif u oni...
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...
3   ham  U dun say so early hor... U c already then say...
4   ham  Nah I don't think he goes to usf, he lives aro...
     v1                                                 v2  label_num
0   ham  Go until jurong point, crazy.. Available only ...          0
1   ham                      Ok lar... Joking wif u oni...          0
2  spam  Free entry in 2 a wkly comp to win FA Cup fina...          1
3   ham  U dun say so early hor... U c already then say...          0
4   ham  Nah I don't think he goes to usf, he lives aro...          0
Accuracy :  0.9856424982053122
Confusion Matrix : 
 [[1203    8]
 [  12  170]]


"""