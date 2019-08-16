import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree

all_data = open(r'C:\Users\Administrator\Desktop\TREE.csv')
rows = csv.reader(all_data)
headers = rows.__next__()
print(headers)

feature_list = []
label_list = []

for row in rows:
    label_list.append(row[len(row) - 1])
    row_dict = {}
    for i in range(1, len(row) - 1):
        row_dict[headers[i]] = row[i]
    feature_list.append(row_dict)

print(feature_list)

vec = DictVectorizer()
dummy_x = vec.fit_transform(feature_list).toarray()
print('dummy_x: ' + str(dummy_x))
print(vec.get_feature_names())
print('label_list: ' + str(label_list))

lb = preprocessing.LabelBinarizer()
dummy_y = lb.fit_transform(label_list)
print('dummy_y: ' + str(dummy_y))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummy_x, dummy_y)
print('clf: ' + str(clf))

with open('abc.dot', 'w') as f:
    tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)


