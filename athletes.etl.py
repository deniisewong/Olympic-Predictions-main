import pandas as pd

# historial dataset scraped by
# https://github.com/chanronnie/Olympics/tree/main
# hosts data from
# https://www.kaggle.com/datasets/piterfm/olympic-games-medals-19862018
# world development indicators  from
# https://data.worldbank.org/

# This script cleans up the historical dataset of olympic athletes and adds socioeconomic data as features


def main():

    # generate athlete info
    hosts = pd.read_csv('data/olympic_hosts.csv')
    hosts = hosts[hosts['game_season'] == 'Summer']
    hosts.drop(columns=['game_slug', 'game_end_date',
                        'game_start_date', 'game_name', 'game_season'], inplace=True)

    # read and clean historical athlete data
    athletes = pd.read_csv('data/athletes.csv')
    athletes.drop(columns=['id', 'died'], inplace=True)
    athletes['height'] = athletes['height'].str.replace(' cm', '')
    athletes['weight'] = athletes['weight'].str.replace(' kg', '')
    athletes['weight'] = athletes['weight'].str.extract(r'(\d{2})')
    athletes = athletes[athletes['game'].str.contains('Summer')]
    athletes['medal'] = athletes['medal'].fillna("None")

    athletes['game_year'] = athletes['game'].str.extract(
        r'(\d{4})').astype(int)

    athletes.dropna(inplace=True)

    athletes['game_year'] = athletes['game_year'].astype(int)
    athletes['height'] = athletes['height'].astype(int)
    athletes['weight'] = athletes['weight'].astype(int)
    athletes['born_year'] = athletes['born'].str.extract(
        r'(\d{4})').astype(int)
    athletes['age'] = athletes['game_year'] - athletes['born_year']
    athletes['age'] = athletes['age'].astype(int)

    athletes = pd.merge(athletes, hosts, on='game_year', how='left')
    athletes['is_host'] = athletes['team'] == athletes['game_location']

    athletes.drop(columns=['born', 'team', 'game',
                  'event', 'born_year', 'game_location'], inplace=True)
    athletes['sport'] = athletes['sport'].str.replace(r'\s*\(.*?\)', '', regex=True)

    athletes = athletes[athletes['weight'] >= 41]

    # read and obtain returning Canadian athletes for paris
    paris_data = pd.read_csv('data/team_canada_paris_2024.csv')
    current_athletes = paris_data.rename(columns={'SPORT (EN)': 'Sport',
                                                  'FIRST NAME / PRÉNOM': 'First Name',
                                                  'LAST NAME / NOM': 'Last Name',
                                                  'AGE (AT GAMES) / ÂGE (AUX JEUX)': 'Age'})

    # Reformat names
    current_athletes['Name'] = current_athletes['First Name'] + \
        ' ' + current_athletes['Last Name']
    current_athletes = current_athletes.drop(['First Name', 'Last Name', 'PRONUNCIATION / PRONONCIATION',
                                              'SPORT (FR)', 'HOMETOWN / LIEU DE RÉSIDENCE', 'HOME PROVINCE / PROVINCE DE RÉSIDENCE', 'LANGUAGES / LANGUES'], axis=1)
    current_athletes = current_athletes.merge(
        athletes, left_on=['Name'], right_on=['name'], how='inner')
    current_athletes['game_year'] = 2024
    current_athletes.drop(columns=['age', 'medal'], inplace=True)
    current_athletes['Sport'] = current_athletes['Sport'].str.replace(r'\s*\(.*?\)', '', regex=True)


    current_athletes.rename(columns={'Age': 'age'}, inplace=True)
    current_athletes = current_athletes[current_athletes['noc'] != 'USA']

    # fetch economic data
    economicData = pd.read_csv('out/economic_data.csv')
    economic_data_melted = pd.melt(
        economicData, id_vars=['Series', 'noc'], var_name='year', value_name='value')
    economic_data_melted['year'] = economic_data_melted['year'].astype(int)

    # merge the economic data with athletes
    merged_data = pd.merge(athletes, economic_data_melted, left_on=[
        'noc', 'game_year'], right_on=['noc', 'year'], how='left')
    merged_data.dropna(inplace=True)
    final_data = merged_data.pivot_table(index=['name', 'gender', 'height', 'weight', 'noc', 'medal', 'game_year', 'age', 'is_host', 'sport'],
                                         columns='Series',
                                         values='value').reset_index()
    final_data.columns.name = None

    final_data.to_csv('out/historical_athletes.csv', index=False)

    # merge economic data for current athletes
    merged_curr_data = pd.merge(current_athletes, economic_data_melted, left_on=[
        'noc', 'game_year'], right_on=['noc', 'year'], how='left')
    merged_curr_data.dropna(inplace=True)
    final_curr_data = merged_curr_data.pivot_table(index=['name', 'gender', 'height', 'weight', 'noc', 'game_year', 'age', 'is_host', 'Sport'],
                                                   columns='Series',
                                                   values='value').reset_index()
    final_curr_data.columns.name = None
    final_curr_data.to_csv('out/returning_athletes.csv', index=False)


if __name__ == '__main__':
    main()
