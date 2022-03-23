#%%
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from functions import formatKenPomData, getModelOutcome


kenpom_data_df = pd.DataFrame()
for i in range(2002,2020):
    kenpom_data = pd.read_excel('data/March madness data.xlsx',sheet_name='Sheet'+str(i-2001),skiprows=1).dropna()
    kenpom_data['Year'] = i
    kenpom_data_df = kenpom_data_df.append(kenpom_data)

kenpom_data_df = formatKenPomData(kenpom_data_df)
kenpom_data_df_losing_team = kenpom_data_df.rename(columns={'Team':'Team.1'})


historical_data = pd.read_csv('data/Big_Dance_CSV.csv')
historical_data = historical_data[historical_data['Year']>2001]

historical_data['Team.1'] = historical_data['Team.1'].str.replace("Cal Irvine",'UC Irvine')

#%%

#create a model for each round via excel sheets
round1_test_og = pd.read_excel('matchups/round1_test_og.xlsx')
round2_test_og = pd.read_excel('matchups/round2_test_og.xlsx')
round3_test_og = pd.read_excel('matchups/round3_test_og.xlsx')
round4_test_og = pd.read_excel('matchups/round4_test_og.xlsx')
round5_test_og = pd.read_excel('matchups/round5_test_og.xlsx')
round6_test_og = pd.read_excel('matchups/round6_test_og.xlsx')
outcome = getModelOutcome(round_num=1,kenpom_data_df_losing_team,historical_data,round1_test_og)
outcome = getModelOutcome(2,kenpom_data_df_losing_team,historical_data,round2_test_og)
outcome = getModelOutcome(3,kenpom_data_df_losing_team,historical_data,round3_test_og)
outcome = getModelOutcome(4,kenpom_data_df_losing_team,historical_data,round3_test_og)
outcome = getModelOutcome(5,kenpom_data_df_losing_team,historical_data,round5_test_og)
outcome = getModelOutcome(6,kenpom_data_df_losing_team,historical_data,round6_test_og)

