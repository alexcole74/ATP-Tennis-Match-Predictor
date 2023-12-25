import pandas as pd

x = 0
df = pd.read_csv ('2022_testing_r.csv')
for index, row in df.iterrows():
    print(row['p1_name'], row['winner_name'])
    if row['p1_name'] == row['winner_name']:
        x = x+1
        df.at[index,'winner_name'] = 1
    else:
        x=x-1
        df.at[index, 'winner_name'] = 0


print("total x value:", x)
print(df)
df.to_csv('2022_testing_r.csv', index=False)