import pandas as pd

df = pd.read_csv("./data/raw/RedditNews.csv", sep=',')

grouped = df.groupby('Date')['News'].apply(list).reset_index()

grouped.to_csv(path_or_buf='./data/interim/Grouped.csv', index=False)
