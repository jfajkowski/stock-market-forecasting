import pandas as pd
import matplotlib.pyplot as plt

ChangeColumn = 'Change'
ClassColumn = 'Class'

#load data
djia = pd.read_csv("./data/raw/DJIA_table.csv", sep=',')
combined = pd.read_csv("./data/raw/Combined_News_DJIA.csv", sep=',')

#calculate relative change in index value
diff = ((djia['Open'] - djia['Close']) / djia['Open'])
djia[ChangeColumn] = diff

#assign new classes
djia.loc[djia[ChangeColumn] < -0.025, ClassColumn] = 0
djia.loc[(djia[ChangeColumn] >= -0.025) & (djia[ChangeColumn] < -0.005), ClassColumn] = 1
djia.loc[(djia[ChangeColumn] >= -0.005) & (djia[ChangeColumn] < 0.005), ClassColumn] = 2
djia.loc[(djia[ChangeColumn] >= 0.005) & (djia[ChangeColumn] < 0.025), ClassColumn] = 3
djia.loc[0.025 <= djia[ChangeColumn], ClassColumn] = 4

#join new classes with combined file
combined = combined.join(djia.set_index('Date')[ClassColumn], on='Date')
combined = combined.drop(['Label'], axis=1)

combined.to_csv(path_or_buf='./data/interim/Outcome.csv', index=False)

# diff = diff.sort_values().reset_index(drop=True)
# diff.plot()
# plt.show()

