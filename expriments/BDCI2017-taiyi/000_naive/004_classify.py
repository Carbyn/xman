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
        y.append(emotions[key] + 3)
    elif key in topics:
        y.append(1)
    else:
        y.append(0)

clf = svm.NuSVC(nu=0.00478)
clf.fit(X, y)

joblib.dump(clf, 'clf.pkl')

nu = 0.005
while nu > 0:
    nu -= 0.00002
    clf = svm.NuSVC(nu=nu, degree=4)
    clf.fit(X, y)
    predict = clf.predict(X)
    error = 0
    for i in xrange(len(y)):
        if y[i] != predict[i]:
            error += 1
    print nu, error
