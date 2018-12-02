import urllib.request

# FastText vectors
urllib.request.urlretrieve('https://s3-us-west-1.amazonaws.com/fasttext-vectors/cc.en.300.vec.gz',
                           './data/external/cc.en.300.vec.gz')