import matplotlib.pyplot as plt
import pandas as pd

ChangeColumn = 'Change'
ClassColumn = 'Class'

# %% Load data
djia = pd.read_csv("./data/raw/DJIA_table.csv", sep=',')
combined = pd.read_csv("./data/raw/Combined_News_DJIA.csv", sep=',')

# %% Calculate relative change in index value
diff = ((djia['Close'] - djia['Open']) / djia['Open'])

# %% Calculate classification boundaries
diff_abs = sorted(abs(diff))
chunk_size = len(diff_abs) / 5
split_point_inner = diff_abs[int(chunk_size)]
split_point_outer = diff_abs[int(3 * chunk_size)]

djia[ChangeColumn] = diff

# %% Assign a class to each sample
djia.loc[djia[ChangeColumn] < -split_point_outer, ClassColumn] = 0
djia.loc[(djia[ChangeColumn] >= -split_point_outer) & (djia[ChangeColumn] < -split_point_inner), ClassColumn] = 1
djia.loc[(djia[ChangeColumn] >= -split_point_inner) & (djia[ChangeColumn] < split_point_inner), ClassColumn] = 2
djia.loc[(djia[ChangeColumn] >= split_point_inner) & (djia[ChangeColumn] < split_point_outer), ClassColumn] = 3
djia.loc[split_point_outer <= djia[ChangeColumn], ClassColumn] = 4

# %% Make new class appear in combined file
combined = combined.join(djia.set_index('Date'), on='Date')
combined = combined.drop(['Label'], axis=1)

# %% Sort by date ascending
combined = combined.sort_values(['Date']).reset_index(drop=True)
combined.to_csv(path_or_buf='./data/interim/Classes_Changed.csv', index=False)

# %% Show stats
print(djia.groupby([ClassColumn]).size())
diff = diff.sort_values().reset_index(drop=True)
diff.plot()
plt.show()



