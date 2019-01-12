import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,

df = pd.read_csv('../scripts/data/data/interim/Classes_Changed.csv', lineterminator='\n', sep=',')
df.columns = df.columns.str.strip()

raw = df.loc[:, 'Top1':'Top25'].apply(lambda x: ' '.join([str(s) for s in x]), axis=1)
y = df['Class']

raw_train, raw_test, y_train, y_test = train_test_split(raw, y, train_size=0.8, shuffle=False)

function_shapes = ['ovr', 'ovo']
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for func, kernel in [(func, kernel) for func in function_shapes for kernel in kernels]:

    clf = SVC(decision_function_shape=func, kernel=kernel)
#clf1 = LinearSVC()
#clf2 = OneVsRestClassifier(LinearSVC(class_weight='balanced'))

    model = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('scale', StandardScaler(with_mean=False)),
        ('clf', clf),
    ])
    model.fit(raw_train, y_train)
    print(func + '/' + kernel + ": " + str(accuracy_score(y_test, model.predict(raw_test))))
