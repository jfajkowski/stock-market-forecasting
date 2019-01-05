import pandas as pd
import matplotlib.pyplot as plt

ChangeColumn = 'Change'
ClassColumn = 'Class'

#load data
djia = pd.read_csv("./data/raw/DJIA_table.csv", sep=',')
combined = pd.read_csv("./data/raw/Combined_News_DJIA.csv", sep=',')

#calculate relative change in index value
diff = ((djia['Close'] - djia['Open']) / djia['Open'])
diff_abs = sorted(abs(diff))
# diff_abs = sorted(diff[diff > 0])

chunk_size = len(diff_abs) / 5
split_point_inner = diff_abs[int(chunk_size)]
split_point_outer = diff_abs[int(3 * chunk_size)]

djia[ChangeColumn] = diff

#assign new classes
djia.loc[djia[ChangeColumn] < -split_point_outer, ClassColumn] = 0
djia.loc[(djia[ChangeColumn] >= -split_point_outer) & (djia[ChangeColumn] < -split_point_inner), ClassColumn] = 1
djia.loc[(djia[ChangeColumn] >= -split_point_inner) & (djia[ChangeColumn] < split_point_inner), ClassColumn] = 2
djia.loc[(djia[ChangeColumn] >= split_point_inner) & (djia[ChangeColumn] < split_point_outer), ClassColumn] = 3
djia.loc[split_point_outer <= djia[ChangeColumn], ClassColumn] = 4

#join new classes with combined file
combined = combined.join(djia.set_index('Date'), on='Date')
combined = combined.drop(['Label'], axis=1)

combined.to_csv(path_or_buf='./data/interim/Classes_Changed.csv', index=False)

print(djia.groupby([ClassColumn]).size())

diff = diff.sort_values().reset_index(drop=True)
diff.plot()
plt.show()



