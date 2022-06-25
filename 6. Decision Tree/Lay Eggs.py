import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, tree
from sklearn.metrics import classification_report, confusion_matrix

w = pd.read_csv("layeggs.csv")
#print(w)
#print(w.describe())

X = w[['Animal', 'Warmblooded', 'Feathers', 'Fur', 'Swims']]
y = w[['Class']]

le = LabelEncoder()
pwnew = pd.DataFrame()
data_top = X.head()
print(data_top)
for val in data_top:
    #print('val : ', val)
    pwnew[val] = le.fit_transform(w[val])
pwnew = pwnew.drop(columns=['Animal'])
print('pwnew : \n', pwnew)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

clf = clf.fit(pwnew, y)

y_pred = clf.predict(pwnew)
print(y_pred)

print("Accuracy : ", metrics.accuracy_score(y, y_pred))
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))

fn = ['Warmblooded', 'Feathers', 'Fur', 'Swims']
cn = ['No', 'Yes']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
tree.plot_tree(clf, feature_names=fn, class_names=cn, filled=True)
fig.savefig('Layeggs.png')

# Tree Drawing, Entropy vs Ginni Indexing, why not use Animal for classification?

"""

      Animal Warmblooded Feathers Fur Swims
0    Ostrich         Yes      Yes  No    No
1  Crocodile          No       No  No   Yes
2      Raven         Yes      Yes  No    No
3  Albatross         Yes      Yes  No    No
4    Dolphin         Yes       No  No   Yes
pwnew : 
    Warmblooded  Feathers  Fur  Swims
0            1         1    0      0
1            0         0    0      1
2            1         1    0      0
3            1         1    0      0
4            1         0    0      1
5            1         0    1      0
['Yes' 'Yes' 'Yes' 'Yes' 'No' 'No']
Accuracy :  1.0
[[2 0]
 [0 4]]
              precision    recall  f1-score   support

          No       1.00      1.00      1.00         2
         Yes       1.00      1.00      1.00         4

    accuracy                           1.00         6
   macro avg       1.00      1.00      1.00         6
weighted avg       1.00      1.00      1.00         6

"""