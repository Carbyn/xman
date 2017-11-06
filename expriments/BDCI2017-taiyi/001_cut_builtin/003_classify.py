"""268"""
import codecs
from sklearn import svm
from sklearn.externals import joblib

vectors = {}
with codecs.open('features', 'r', 'utf8') as reader:
    for line in reader:
        values = line.strip().split(' ')
        word = values[0]
        vector = map(float, values[1:])
        vectors[word] = vector

topics = {}
with codecs.open('topics', 'r', 'utf8') as reader:
    for line in reader:
        topics[line.strip()] = True

emotions = {}
with codecs.open('emotions', 'r', 'utf8') as reader:
    for line in reader:
        word, rank = line.strip().split(' ')
        emotions[word] = int(rank)

X, y = [], []
for key, val in vectors.items():
    X.append(val)
    if key in emotions:
        y.append(2)
    elif key in topics:
        y.append(1)
    else:
        y.append(0)

clf = svm.NuSVC(nu=0.005)
clf.fit(X, y)

joblib.dump(clf, 'attr_clf.pkl')

X, y = [], []
for key, val in vectors.items():
    if key in emotions:
        X.append(val)
        y.append(emotions[key])

clf = svm.NuSVC(nu=0.005)
clf.fit(X, y)

joblib.dump(clf, 'rank_clf.pkl')
