import pandas as pd

def formatKenPomData(kenpom_data_df):
    #this is done to clean the dataset
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace('\d+', '')
    kenpom_data_df['NCSOS'] = kenpom_data_df['NCSOS'].astype('str')
    kenpom_data_df['SOS'] = kenpom_data_df['SOS'].astype('str')
    kenpom_data_df['Luck'] = kenpom_data_df['Luck'].astype('str')
    kenpom_data_df['AdjEM'] = kenpom_data_df['AdjEM'].astype('str')


    kenpom_data_df['SOS'] = kenpom_data_df['SOS'].str.replace("+","")
    kenpom_data_df['NCSOS'] = kenpom_data_df['NCSOS'].str.replace("+","")
    kenpom_data_df['Luck'] = kenpom_data_df['Luck'].str.replace("+","")
    kenpom_data_df['AdjEM'] = kenpom_data_df['AdjEM'].str.replace("+","")


    #manual data conditioning to combine datasets
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Central Connecticut","Central Connecticut St")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Penn","Pennsylvania")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace(".","")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("UC Santa Barbara","Santa Barbara")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Miami FL","Miami")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Miami OH","Miami Ohio")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Mississippi","Ole Miss")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("N.C. Stte","NC State")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Stnford","Stanford")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("John's","Johns")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Troy St","Troy")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Saint Joseph's","St Josephs")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Ole Miss St","Mississippi St")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("UCF","Central Florida")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("UTSA","Texas San Antonio")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Saint Mary's","St Marys")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Texas A&M Corpus Chris","Texas A&M Corpus Christi")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("UT Arlington","Texas Arlington")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Saint","St")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Peter's","Peters")
    kenpom_data_df['Team'] = kenpom_data_df['Team'].str.replace("Milwaukee","Wisconsin Milwaukee")
    return kenpom_data_df

def getModelOutcome(round_num,kenpom_data_df_losing_team,historical_data,round_df_test_og):
    round_df = historical_data[historical_data['Round']==round_num]

    #append winning team
    round_df = round_df.merge(kenpom_data_df,on=['Year','Team'])

    #append losing team
    round_df = round_df.merge(kenpom_data_df_losing_team,on=['Year','Team.1'])
    #round1.to_excel('round1.xlsx')

    round_df['Conf_x'] = round_df['Conf_x']+'_x'
    round_df['Conf_y'] = round_df['Conf_y']+'_y'

    data = pd.concat((round_df,pd.get_dummies(round_df['Conf_x']),pd.get_dummies(round_df['Conf_y'])),axis=1)

    data = data.drop(['Conf_x','Conf_y'],axis=1)

    for i in range(len(data)):
        if data.loc[i,'Score'] > data.loc[i,'Score.1']:
            data.loc[i,'Won?'] = 'x'
        else:
            data.loc[i,'Won?'] = 'y'
    y = data['Won?']
    x = data.drop(['Region Name','Round','Team','Team.1','Won?'],axis=1)

    for i in range(len(x)):
        wins = float(x['W-L_x'][i].split('-')[0])
        losses = float(x['W-L_x'][i].split('-')[1])
        total = wins + losses
        x['W-L_x'][i] = wins/total
        wins = float(x['W-L_y'][i].split('-')[0])
        losses = float(x['W-L_y'][i].split('-')[1])
        total = wins + losses
        x['W-L_y'][i] = wins/total
    x = x.astype('float')


    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        shuffle = True, 
                                                        test_size=0.1, 
                                                        random_state=1)

    baseline_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')



    baseline_model.fit(x,y)
    y_predicted = baseline_model.predict(X_test)
    y_joined = pd.concat((y_test,pd.DataFrame(y_predicted,index=y_test.index)),axis=1)

    y_joined['Same'] = y_joined['Won?'] == y_joined[0]
    pctright = y_joined['Same'].sum()/len(y_joined)

    kenpom_data_df_test = pd.read_excel('March madness_2022.xlsx',skiprows=1,sheet_name='Sheet20')
    kenpom_data_df_test = formatKenPomData(kenpom_data_df_test)
    kenpom_data_df_test['Year'] = 2022
    kenpom_data_df_losing_team = kenpom_data_df_test.rename(columns={'Team':'Team.1'})
    round_df_test = round_df_test_og.merge(kenpom_data_df_test,on=['Year','Team'])
    round_df_test = round_df_test.merge(kenpom_data_df_losing_team,on=['Year','Team.1'])
    round_df_test['Conf_x'] = round_df_test['Conf_x']+'_x'
    round_df_test['Conf_y'] = round_df_test['Conf_y']+'_y'
    round_df_test = pd.concat((round_df_test,pd.get_dummies(round_df_test['Conf_x']),pd.get_dummies(round_df_test['Conf_y'])),axis=1)
    for i in range(len(round_df_test)):
        wins = float(round_df_test['W-L_x'][i].split('-')[0])
        losses = float(round_df_test['W-L_x'][i].split('-')[1])
        total = wins + losses
        round_df_test['W-L_x'][i] = wins/total
        wins = float(round_df_test['W-L_y'][i].split('-')[0])
        losses = float(round_df_test['W-L_y'][i].split('-')[1])
        total = wins + losses
        round_df_test['W-L_y'][i] = wins/total
    for col in x.columns[~x.columns.isin(list(round_df_test.columns))]:
        round_df_test[col] = 0

    round_df_test = round_df_test.drop(['Team','Team.1','Conf_x','Conf_y'],axis=1)
    round_df_test = round_df_test.astype('float')
    y_predicted_new = baseline_model.predict(round_df_test)

    outcome = pd.concat((round_df_test_og,pd.DataFrame(y_predicted_new,index=round_df_test_og.index)),axis=1)
    return outcome

