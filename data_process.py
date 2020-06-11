import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# configure
DIR = "data/MDataFiles_Stage1/"
file_tourney_seed = 'basic/MNCAATourneySeeds.csv'
file_tourney_history = 'detail/MNCAATourneyDetailedResults.csv'
file_regular_history = 'detail/MRegularSeasonDetailedResults.csv'
file_submission = 'data/MSampleSubmissionStage1_2020.csv'
file_teams = 'basic/MTeams.csv'


def load_data():
    df_submission = pd.read_csv(file_submission)
    df_teams = pd.read_csv('{}/{}'.format(DIR, file_teams))
    df_tourney_seeds = pd.read_csv('{}/{}'.format(DIR, file_tourney_seed))
    df_tourney_history = pd.read_csv('{}/{}'.format(DIR, file_tourney_history))
    df_regular_history = pd.read_csv('{}/{}'.format(DIR, file_regular_history))
    return df_submission, df_teams, df_tourney_seeds, df_tourney_history, df_regular_history


def prepare_data(df_tourney_seeds, df_tourney_history, df_regular_history, saved=False,norm=True):
    df_regular_history['Is_Tournament_Match'] = 0
    df_tourney_history['Is_Tournament_Match'] = 1
    df_history = pd.concat([
        df_regular_history,
        df_tourney_history
    ])
    df_winning_teams = df_history[[
        'Season',
        'DayNum',
        'WTeamID',
        'LTeamID',
        'WScore',
        'LScore',
        'WLoc',
        'WFGM',
        'WFGA',
        'WFGM3',
        'WFGA3',
        'WFTM',
        'WFTA',
        'WOR',
        'WDR',
        'WAst',
        'WTO',
        'WStl',
        'WBlk',
        'WPF',
        'LFGM',
        'LFGA',
        'LFGM3',
        'LFGA3',
        'LFTM',
        'LFTA',
        'LOR',
        'LDR',
        'LAst',
        'LTO',
        'LStl',
        'LBlk',
        'LPF',
        'Is_Tournament_Match'
    ]].rename(columns={
        'WTeamID': 'TeamID',
        'LTeamID': 'TeamID_OPP',
        'WScore': 'PTS',
        'LScore': 'PTS_OPP',
        'WLoc': 'Location',
        'WFGM': 'FGM',
        'LFGM': 'FGM_OPP',
        'WFGA': 'FGA',
        'LFGA': 'FGA_OPP',
        'WFGM3': 'FG3M',
        'LFGM3': 'FG3M_OPP',
        'WFGA3': 'FG3A',
        'LFGA3': 'FG3A_OPP',
        'WFTM': 'FTM',
        'LFTM': 'FTM_OPP',
        'WFTA': 'FTA',
        'LFTA': 'FTA_OPP',
        'WFTM': 'FTM',
        'LFTM': 'FTM_OPP',
        'WOR': 'OREB',
        'LOR': 'OREB_OPP',
        'WDR': 'DREB',
        'LDR': 'DREB_OPP',
        'WAst': 'AST',
        'LAst': 'AST_OPP',
        'WTO': 'TO',
        'LTO': 'TO_OPP',
        'WStl': 'STL',
        'LStl': 'STL_OPP',
        'WBlk': 'BLK',
        'LBlk': 'BLK_OPP',
        'WPF': 'PF',
        'LPF': 'PF_OPP',
    })
    df_losing_teams = df_history[[
        'Season',
        'DayNum',
        'WTeamID',
        'LTeamID',
        'WScore',
        'LScore',
        'WLoc',
        'WFGM',
        'WFGA',
        'WFGM3',
        'WFGA3',
        'WFTM',
        'WFTA',
        'WOR',
        'WDR',
        'WAst',
        'WTO',
        'WStl',
        'WBlk',
        'WPF',
        'LFGM',
        'LFGA',
        'LFGM3',
        'LFGA3',
        'LFTM',
        'LFTA',
        'LOR',
        'LDR',
        'LAst',
        'LTO',
        'LStl',
        'LBlk',
        'LPF',
        'Is_Tournament_Match'
    ]].rename(columns={
        'LTeamID': 'TeamID',
        'WTeamID': 'TeamID_OPP',
        'LScore': 'PTS',
        'WScore': 'PTS_OPP',
        'LFGM': 'FGM',
        'WFGM': 'FGM_OPP',
        'LFGA': 'FGA',
        'WFGA': 'FGA_OPP',
        'LFGM3': 'FG3M',
        'WFGM3': 'FG3M_OPP',
        'LFGA3': 'FG3A',
        'WFGA3': 'FG3A_OPP',
        'LFTM': 'FTM',
        'WFTM': 'FTM_OPP',
        'LFTA': 'FTA',
        'WFTA': 'FTA_OPP',
        'LFTM': 'FTM',
        'WFTM': 'FTM_OPP',
        'LOR': 'OREB',
        'WOR': 'OREB_OPP',
        'LDR': 'DREB',
        'WDR': 'DREB_OPP',
        'LAst': 'AST',
        'WAst': 'AST_OPP',
        'LTO': 'TO',
        'WTO': 'TO_OPP',
        'LStl': 'STL',
        'WStl': 'STL_OPP',
        'LBlk': 'BLK',
        'WBlk': 'BLK_OPP',
        'LPF': 'PF',
        'WPF': 'PF_OPP',
    })
    df_losing_teams['Location'] = df_losing_teams.apply(
        lambda x: 'A' if x['WLoc'] == 'H' else 'H' if x['WLoc'] == 'A' else 'N',
        axis=1
    )
    df_teams = pd.concat([
        df_winning_teams,
        df_losing_teams
    ])[[
        'Season',
        'DayNum',
        'Location',
        'TeamID',
        'PTS',
        'FGM', 'FGA',
        'FG3M', 'FG3A',
        'FTM', 'FTA',
        'AST',
        'OREB', 'DREB',
        'STL', 'BLK',
        'TO',
        'PF',
        'TeamID_OPP',
        'PTS_OPP',
        'FGM_OPP', 'FGA_OPP',
        'FG3M_OPP', 'FG3A_OPP',
        'FTM_OPP', 'FTA_OPP',
        'AST_OPP',
        'OREB_OPP', 'DREB_OPP',
        'STL_OPP', 'BLK_OPP',
        'TO_OPP',
        'PF_OPP',
        'Is_Tournament_Match'
    ]]
    df_teams['FG2M'] = df_teams['FGM'] - df_teams['FG3M']
    df_teams['FG2A'] = df_teams['FGA'] - df_teams['FG3A']
    df_teams['FG2_PCT'] = df_teams['FG2M'] / df_teams['FG2A']
    df_teams['FG3_PCT'] = df_teams['FG3M'] / df_teams['FG3A']
    df_teams['FG_PCT'] = df_teams['FGM'] / df_teams['FGA']
    df_teams['FT_PCT'] = (df_teams['FTM'] / df_teams['FTA']).fillna(0.691749)
    df_teams['FG2M_OPP'] = df_teams['FGM_OPP'] - df_teams['FG3M_OPP']
    df_teams['FG2A_OPP'] = df_teams['FGA_OPP'] - df_teams['FG3A_OPP']
    df_teams['FG2_PCT_OPP'] = df_teams['FG2M_OPP'] / df_teams['FG2A_OPP']
    df_teams['FG3_PCT_OPP'] = df_teams['FG3M_OPP'] / df_teams['FG3A_OPP']
    df_teams['FG_PCT_OPP'] = df_teams['FGM_OPP'] / df_teams['FGA_OPP']
    df_teams['FT_PCT_OPP'] = (df_teams['FTM_OPP'] / df_teams['FTA_OPP']).fillna(0.691749)
    df_teams['PTS_SPREAD'] = df_teams['PTS'] - df_teams['PTS_OPP']
    df_teams['IND_LOCATION_HOME'] = df_teams.apply(
        lambda x: 1 if x['Location'] == 'H' else 0,
        axis=1
    )
    df_teams['IND_LOCATION_NEUTRAL'] = df_teams.apply(
        lambda x: 1 if x['Location'] == 'N' else 0,
        axis=1
    )
    df_teams['IND_LOCATION_AWAY'] = df_teams.apply(
        lambda x: 1 if x['Location'] == 'A' else 0,
        axis=1
    )
    df_teams['IND_LOCATION_HOME_OPP'] = df_teams.apply(
        lambda x: 0 if x['Location'] == 'H' else 1,
        axis=1
    )
    df_teams['IND_LOCATION_NEUTRAL_OPP'] = df_teams.apply(
        lambda x: 1 if x['Location'] == 'N' else 0,
        axis=1
    )
    df_teams['IND_LOCATION_AWAY_OPP'] = df_teams.apply(
        lambda x: 0 if x['Location'] == 'A' else 1,
        axis=1
    )
    # df_teams['Location_OPP'] = df_teams.apply(
    #     lambda x: 'H' if x['Location'] == 'A' else 'A' if x['Location'] == 'H' else 'N',
    #     axis=1
    # )
    df_teams = df_teams.merge(
        df_tourney_seeds,
        on=['TeamID', 'Season'],
        how='left'
    ).merge(
        df_tourney_seeds.rename(
            columns={
                'TeamID': 'TeamID_OPP',
                'Seed': 'Seed_OPP'
            }),
        on=['TeamID_OPP', 'Season'],
        how='left'
    )
    df_teams['Seed'] = df_teams['Seed'].apply(
        lambda x: 8 if x is np.NaN else int(x[1:-1]) if not x[-1:].isdigit() else int(x[1:])
    )
    df_teams['Seed_OPP'] = df_teams['Seed_OPP'].apply(
        lambda x: 8 if x is np.NaN else int(x[1:-1]) if not x[-1:].isdigit() else int(x[1:])
    )
    df_teams.drop(['Location','PTS','PTS_OPP',  'Is_Tournament_Match','DayNum'], axis=1, inplace=True)
    df_teams = df_teams.sample(frac=1).reset_index(drop=True)
    # get X and Y
    Y = df_teams[['PTS_SPREAD']]
    df_teams.drop('PTS_SPREAD', axis=1, inplace=True)

    # # fullfil missing free throw
    # mean_ftp = df_teams[['FT_PCT']].mean(skipna=True)

    # normalization
    data_norm=df_teams
    if saved:
        data_norm.to_csv('cleaned_data/data.csv', index=False)
        data_norm = normalize(data_norm)
        data_norm.to_csv('cleaned_data/unnom_data.csv', index=False)
        Y.to_csv('cleaned_data/result.csv', index=False)
        Y['bin'] = Y['PTS_SPREAD'].apply(
            lambda x: 1 if x >= 0 else 0
        )
        Y.drop('PTS_SPREAD', axis=1, inplace=True)
        Y.to_csv('cleaned_data/binary.csv', index=False)
    return data_norm, Y.values


# pca
# component: number of features
# data: numpy type
def transform_pca(data, component):
    pca = PCA(n_components=component)
    pca.fit(data)
    return pca.transform(data)


def retain_pca(data, retained_var=0.99):
    D = data.shape[1]
    pca = PCA(n_components=D)
    pca.fit(data)
    # pca.fit_transform()
    var = pca.explained_variance_ratio_
    cur_var = 0
    comp = 0
    while comp<D:
        cur_var += var[comp]
        if cur_var > retained_var:
            break
        comp+=1
    return transform_pca(data,comp)


# min max normalization
def normalize(data, scale=20):
    data_norm = (data - data.min()) / (data.max() - data.min())
    if scale > 0:
        data_norm *= scale
    return data_norm


# get data
def get_data_from_csv():
    data = pd.read_csv('cleaned_data/data.csv')
    result = pd.read_csv('cleaned_data/result.csv')
    binary = pd.read_csv('cleaned_data/binary.csv')
    return data.values, result.values, binary.values


def get_data():
    df_submission, df_teams, df_tourney_seeds, df_tourney_history, df_regular_history = load_data()
    prepare_data(df_tourney_seeds, df_tourney_history, df_regular_history, saved=True)


def get_submission():
    sub = pd.read_csv('data/MSampleSubmissionStage1_2020.csv')
    sub['Season'] = sub['ID'].map(lambda  x:int(x[:4]))
    sub['TeamID'] = sub['ID'].map(lambda x:x[5:9]).astype(int)
    sub['TeamID_OPP'] = sub['ID'].map(lambda x:x[10:14]).astype(int)
    ori_data = pd.read_csv('cleaned_data/unnom_data.csv')
    print(ori_data.head())
    cur_cols = [ 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM',
       'FTA', 'AST', 'OREB', 'DREB', 'STL', 'BLK', 'TO', 'PF', 'FG2M', 'FG2A',
       'FG2_PCT', 'FG3_PCT', 'FG_PCT', 'FT_PCT', 'IND_LOCATION_HOME', 'IND_LOCATION_NEUTRAL',
       'IND_LOCATION_AWAY', 'Seed']
    opp_cols =['FGM_OPP', 'FGA_OPP', 'FG3M_OPP', 'FG3A_OPP', 'FTM_OPP', 'FTA_OPP',
       'AST_OPP', 'OREB_OPP', 'DREB_OPP', 'STL_OPP', 'BLK_OPP', 'TO_OPP',
       'PF_OPP', 'FG2M_OPP', 'FG2A_OPP',
       'FG2_PCT_OPP', 'FG3_PCT_OPP',
       'FG_PCT_OPP', 'FT_PCT_OPP', 'IND_LOCATION_HOME_OPP', 'IND_LOCATION_NEUTRAL_OPP',
       'IND_LOCATION_AWAY_OPP',  'Seed_OPP']
    stat_cur = ori_data.groupby(['Season','TeamID'])[cur_cols].agg([np.mean]).reset_index()
    stat_cur.columns =[''.join(col).strip() for col in stat_cur.columns.values]
    print(stat_cur.columns)
    stat_opp = ori_data.groupby(['Season','TeamID_OPP'])[opp_cols].agg([np.mean]).reset_index()
    stat_opp.columns =[''.join(col).strip() for col in stat_opp.columns.values]
    print(stat_opp.columns)
    sub = pd.merge(sub,stat_cur,on=['Season','TeamID'])
    sub = pd.merge(sub,stat_opp,on=['Season','TeamID_OPP'])
    sub.drop(['ID','Pred'],axis=1, inplace=True)
    sub=sub[[
        'Season',
        'TeamID',
        'FGMmean', 'FGAmean', 'FG3Mmean',
        'FG3Amean', 'FTMmean', 'FTAmean', 'ASTmean', 'OREBmean', 'DREBmean',
        'STLmean', 'BLKmean', 'TOmean', 'PFmean',
        'TeamID_OPP',
        'FGM_OPPmean', 'FGA_OPPmean',
        'FG3M_OPPmean', 'FG3A_OPPmean', 'FTM_OPPmean', 'FTA_OPPmean',
        'AST_OPPmean', 'OREB_OPPmean', 'DREB_OPPmean', 'STL_OPPmean',
        'BLK_OPPmean', 'TO_OPPmean', 'PF_OPPmean',
        'FG2Mmean', 'FG2Amean',
        'FG2_PCTmean', 'FG3_PCTmean', 'FG_PCTmean', 'FT_PCTmean',
        'FG2M_OPPmean',
        'FG2A_OPPmean', 'FG2_PCT_OPPmean', 'FG3_PCT_OPPmean', 'FG_PCT_OPPmean',
        'FT_PCT_OPPmean',
        'IND_LOCATION_HOMEmean', 'IND_LOCATION_NEUTRALmean',
        'IND_LOCATION_AWAYmean',
        'IND_LOCATION_HOME_OPPmean',
        'IND_LOCATION_NEUTRAL_OPPmean', 'IND_LOCATION_AWAY_OPPmean',
        'Seedmean','Seed_OPPmean'
    ]]
    return sub


if __name__ == '__main__':
    df_submission, df_teams, df_tourney_seeds, df_tourney_history, df_regular_history =load_data()
    prepare_data(df_tourney_seeds,df_tourney_history,df_regular_history,True,norm=False)
    get_data_from_csv()
    # get_submission()
