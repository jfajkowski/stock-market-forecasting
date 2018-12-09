import re
import pandas as pd


def strip(s):
    groups = re.match(r'^b\'(.+?)\'$|^b\"(.+?)\"$|(.+)', s.replace('\n', '')).groups()
    return next(g for g in groups if g is not None)


df = pd.read_csv('./data/raw/Combined_News_DJIA.csv')
df.loc[:, 'Top1':'Top25'] = df.loc[:, 'Top1':'Top25'].applymap(lambda x: strip(str(x)))
df.to_csv(path_or_buf='./data/processed/Stripped.csv', index=False)
