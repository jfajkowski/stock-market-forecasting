import urllib.request

# FastText vectors
urllib.request.urlretrieve('https://s3-us-west-1.amazonaws.com/fasttext-vectors/cc.en.300.vec.gz',
                           './data/external/cc.en.300.vec.gz')

# Word2Vec vectors
urllib.request.urlretrieve('https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing',
                           './data/external/GoogleNews-vectors-negative300.bin')