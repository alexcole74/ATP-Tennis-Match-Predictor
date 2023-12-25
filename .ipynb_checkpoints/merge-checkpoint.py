import numpy as np
import pandas as pd
import glob


pd.set_option("display.max_rows", 2000)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)

#get the names of all csv files and print to test
df = pd.read_csv ('2015_2021_training_r.csv')
#print(csv_files)

#concatenate all csv files into one dataframe and print to test
#df_csv_concat = pd.concat([pd.read_csv(file) for file in csv_files ], ignore_index=True)
#print(df_csv_concat)


#make csv from dataframe
#df_csv_concat.to_csv('Data2015_2021_training.csv', index=False)

#groupby to see stats of players such as average loser rank points
#print(df['p1_name'],'\n', df ['p1_id'])


#switch all of the p1 and p2 variables randomly
#np.random.seed(42)
#idx = np.random.rand(len(df)) < 0.5

#df.loc[idx, ['p1_id','p2_id']] = df.loc[idx, ['p2_id','p1_id']].to_numpy()
#df.loc[idx, ['p1_seed','p2_seed']] = df.loc[idx, ['p2_seed','p1_seed']].to_numpy()
#df.loc[idx, ['p1_entry','p2_entry']] = df.loc[idx, ['p2_entry','p1_entry']].to_numpy()
#df.loc[idx, ['p1_name','p2_name']] = df.loc[idx, ['p2_name','p1_name']].to_numpy()
#df.loc[idx, ['p1_hand','p2_hand']] = df.loc[idx, ['p2_hand','p1_hand']].to_numpy()
#df.loc[idx, ['p1_ht','p2_ht']] = df.loc[idx, ['p2_ht','p1_ht']].to_numpy()
#df.loc[idx, ['p1_ioc','p2_ioc']] = df.loc[idx, ['p2_ioc','p1_ioc']].to_numpy()
#df.loc[idx, ['p1_age','p2_age']] = df.loc[idx, ['p2_age','p1_age']].to_numpy()
#df.loc[idx, ['p1_ace','p2_ace']] = df.loc[idx, ['p2_ace','p1_ace']].to_numpy()
#df.loc[idx, ['p1_df','p2_df']] = df.loc[idx, ['p2_df','p1_df']].to_numpy()
#df.loc[idx, ['p1_svpt','p2_svpt']] = df.loc[idx, ['p2_svpt','p1_svpt']].to_numpy()
#df.loc[idx, ['p1_1stIn','p2_1stIn']] = df.loc[idx, ['p2_1stIn','p1_1stIn']].to_numpy()
#df.loc[idx, ['p1_1stWon','p2_1stWon']] = df.loc[idx, ['p2_1stWon','p1_1stWon']].to_numpy()
#df.loc[idx, ['p1_2ndWon','p2_2ndWon']] = df.loc[idx, ['p2_2ndWon','p1_2ndWon']].to_numpy()
#df.loc[idx, ['p1_SvGms','p2_SvGms']] = df.loc[idx, ['p2_SvGms','p1_SvGms']].to_numpy()
#df.loc[idx, ['p1_bpSaved','p2_bpSaved']] = df.loc[idx, ['p2_bpSaved','p1_bpSaved']].to_numpy()
#df.loc[idx, ['p1_bpFaced','p2_bpFaced']] = df.loc[idx, ['p2_bpFaced','p1_bpFaced']].to_numpy()
#df.loc[idx, ['p1_rank','p2_rank']] = df.loc[idx, ['p2_rank','p1_rank']].to_numpy()
#df.loc[idx, ['p1_rank_points','p2_rank_points']] = df.loc[idx, ['p2_rank_points','p1_rank_points']].to_numpy()

#print('\n','\n','\n')
#print(df['p1_name'],'\n', df ['p1_id'])


#df.to_csv('AllData_r.csv', index=False)
#print(df.groupby(['p1_name', 'surface']).agg({'p2_rank': ['mean', 'min', 'max']}))


print(df.isnull().sum())
