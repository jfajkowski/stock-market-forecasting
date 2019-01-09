import json
import pickle
import sys

with open('./models/raw_bigram_linear.vcr', mode='rb') as vcr_file:
    vectorizer = pickle.load(vcr_file)
with open('./models/raw_bigram_linear.clf', mode='rb') as clf_file:
    classifier = pickle.load(clf_file)

# model_input_line = '{"indexHistory":[{"date":"2016-06-30","openValue":2.0,"closeValue":3.0},{"date":"2016-07-01","openValue":1.0,"closeValue":2.0}],"articles":[{"date":"2016-06-29","header":"Test header 3"},{"date":"2016-06-30","header":"Test header 2"},{"date":"2016-07-01","header":"Test header 1"}]}'
model_input_line = next(sys.stdin)
model_input = json.loads(model_input_line)

articles = sorted(model_input['articles'], key=lambda x: x['date'])

raw = list(map(lambda x: x['header'], articles))
X = vectorizer.transform([raw[-1]])
y = classifier.predict_proba(X)
print(y[0])