import random
import requests
import yaml
import time
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

yaml_file = 'champ_select_project/secrets.yml'
with open(yaml_file, 'r') as file:
    yaml_dict = yaml.safe_load(file)
API_KEY = yaml_dict['app_key']
DATABASE_URL = yaml_dict['database_url']


def connect_to_database(yaml_dict, server, database):
    """
    create engine used for connecting to a postgresql database.
    (an engine does not open the connection but we can use it to open connections later)
    :param yaml_dict: dictionary of parsed yaml file
    :param server: string, target server to connect to
    :param database: string, target database to connect to
    :return: engine that will allow
    """
    username = yaml_dict[server][database]['username']
    password = yaml_dict[server][database]['password']
    host = yaml_dict[server][database]['host']
    port = yaml_dict[server][database]['port']
    database_name = yaml_dict[server][database]['database_name']

    connection_url = f'postgresql://{username}:{password}@{host}:{port}/{database_name}'
    try:
        # Create the engine and connect to the database
        engine = create_engine(connection_url)
        # only want to connect when we are running the query and then swiftly disconnect
        # connection = engine.connect()

        # Return the connection object
        # return connection
        return engine

    # this might not do anything anymore once stopped connecting in the try
    except Exception as e:
        print(f"Error connecting to the database: {str(e)}")


def execute_query(database_connection, query):
    """
    uses a engine to open a connection to a database, runs a query. and then closes the connection.
    returning the results of the query in a dataframe (metadata is attached but not in the df)
    :param database_connection: engine, actually an engine that will allow us to open a connection
    :param query: string, query to run against a database
    :return: dataframe of resultant query
    """
    query = text(query)
    database_connection = database_connection.connect()
    query_result = database_connection.execute(query)
    df = pd.DataFrame(query_result.fetchall(), columns=query_result.keys())
    database_connection.close()
    return df

def close_all_active_connections():
    """
    opens a new connection, executes query to close all open connections, closes self at end
    :return: None
    """
    query= """
    SELECT pg_terminate_backend(pg_stat_activity.pid)
    FROM pg_stat_activity
    WHERE pg_stat_activity.datname = 'league_db_server'
    AND pid <> pg_backend_pid();
    """
    db_connection = connect_to_database(yaml_dict, 'league_db_server', 'test_league_db')
    execute_query(db_connection, query)
    return

def find_open_connections():
    """
    creates a new connection lists this open connection as well as any others that are running
    :return:
    """
    query = """
    SELECT *
    FROM pg_stat_activity
    WHERE state = 'active'
"""
    db_connection = connect_to_database(yaml_dict, 'league_db_server', 'test_league_db')
    result = execute_query(db_connection, query)
    return


###############################   Test Code for data base connections    #############################################
######################################################################################################################
# TODO write another file that explains how to use function with examples
# Example how to pull data from database
test_connection = connect_to_database(yaml_dict, 'league_db_server', 'test_league_db')
query = "SELECT * FROM sample_table"
test = execute_query(test_connection, query)
find_open_connections()
close_all_active_connections()
####################################################################################################################


def fetch_api_call(api_url):
    response = requests.get(api_url)
    # Handle rate limit (429) errors by waiting and retrying after a delay
    attempts = 0
    while response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 120))
        current_time = datetime.now()
        print(f'Waiting due to rate limit... Current time: {current_time}')
        time.sleep(retry_after)
        response = requests.get(api_url)
    # Parse the JSON response into match data and store it in the dictionary
    data = response.json()
    return data


def get_leagues(api_key=API_KEY, region='na1', queue_type='RANKED_SOLO_5x5', tier='DIAMOND', divison='I', page='1'):
    """
    :param api_key:
    :param region:
    :param queue_type:
    :param tier:
    :param divison:
    :param page:
    :return:
    """
    paths = {
        'challenger': f'https://{region}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/{queue_type}?api_key={api_key}',
        'grandmaster': f'https://{region}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/{queue_type}?api_key={api_key}',
        'master': f'https://{region}.api.riotgames.com/lol/league/v4/{tier}leagues/by-queue/{queue_type}?api_key={api_key}'
    }
    api_url = paths.get(tier, f'https://na1.api.riotgames.com/lol/league/v4/entries/{queue_type}/{tier}/{divison}?page={page}&api_key={api_key}')
    data = fetch_api_call(api_url)
    return data


################################ START Testing Pulling Match Ids ###################################################
####################################################################################################################

# # read the puuid database table if it exists
# # convert it to dataframe
# # call the new data from riot api
# # convert it to dataframe
# # merge to find only new puuids
# # add these entries to the table
#
#
# from_api = get_leagues(api_key=API_KEY, region='na1', queue_type='RANKED_SOLO_5x5', tier='DIAMOND', divison='I', page='1')
# from_api = pd.DataFrame(from_api)
# from_api = from_api.drop('miniSeries', axis=1)
# if 'miniSeries' in from_api.columns:
#     from_api = from_api.drop('miniSeries', axis=1)
#
# # reads from database table
# db_conn = connect_to_database(yaml_dict, "league_db_server", "test_league_db")
# query = "SELECT * FROM players_tbl"
# from_db = execute_query(db_conn, query)
#
# # use this as an example of things I can do with database
# # finds all rows not already in the database
# merged = from_api.merge(from_db, on='summonerId', how='left', indicator=True, suffixes=('', '_df2'))
# # remove rows that are found in right table
# rows_only_in_api = merged[merged['_merge'] == 'left_only']
# # remove extra set of columns
# start_index = rows_only_in_api.columns.get_loc('leagueId_df2')
# end_index = rows_only_in_api.columns.get_loc('_merge')
# rows_only_in_api = rows_only_in_api.drop(rows_only_in_api.columns[start_index:end_index+1], axis=1)
#
# # add these new rows to the database table
# rows_only_in_api.to_sql('players_tbl', db_conn, if_exists='append', index=False)

####################################################################################################################
############################## END Testing Pulling Match Ids##########################################################


def create_player_list():
    """

    :return: nothing only updates database
    """

    db_conn = connect_to_database(yaml_dict, "league_db_server", "test_league_db")
    query = "SELECT * FROM players_tbl"
    from_db = execute_query(db_conn, query)

    list_of_elo_brackets = [['DIAMOND', 'II'], ['DIAMOND', 'I'], ['master', 'I'], ['grandmaster', 'I'], ['challenger', 'I']]
    for elo_bracket in list_of_elo_brackets:
        if elo_bracket[0] == 'DIAMOND':
            for page in range(1, 51):
                from_api = get_leagues(api_key=API_KEY, region='na1', queue_type='RANKED_SOLO_5x5', tier=elo_bracket[0], divison=elo_bracket[1], page=str(page))
                # ran out of pages for elo bracket
                if len(from_api) == 0:
                    break
                from_api = pd.DataFrame(from_api)
                if 'miniSeries' in from_api.columns:
                    from_api = from_api.drop('miniSeries', axis=1)
                # FOR DIAMOND MERGE EACH PAGE AND ADD TO DATABASE
                merged = from_api.merge(from_db, on='summonerId', how='left', indicator=True, suffixes=('', '_df2'))
                rows_only_in_api = merged[merged['_merge'] == 'left_only']
                start_index = rows_only_in_api.columns.get_loc('leagueId_df2')
                end_index = rows_only_in_api.columns.get_loc('_merge')
                rows_only_in_api = rows_only_in_api.drop(rows_only_in_api.columns[start_index:end_index + 1], axis=1)
                # Save to db
                rows_only_in_api.to_sql('players_tbl', db_conn, if_exists='append', index=False)
        else:
            from_api = get_leagues(api_key=API_KEY, region='na1', queue_type='RANKED_SOLO_5x5', tier=elo_bracket[0], divison=elo_bracket[1], page='1')
            from_api = pd.DataFrame(from_api)
            # from_api = from_api.drop('miniSeries', axis=1)

            # expand entries dictionary
            expanded_data = from_api['entries'].apply(pd.Series)
            from_api = pd.concat([from_api, expanded_data], axis=1)
            from_api = from_api.drop(['entries', 'name'], axis=1)
            from_api = from_api.rename(columns={'queue': 'queueType'})
            # check column are equal for both dataframes
            tier_column = from_api.pop('tier')
            from_api.insert(2, 'tier', tier_column)
            rank_column = from_api.pop('rank')
            from_api.insert(3, 'rank', rank_column)

            merged = from_api.merge(from_db, on='summonerId', how='left', indicator=True, suffixes=('', '_df2'))
            rows_only_in_api = merged[merged['_merge'] == 'left_only']
            start_index = rows_only_in_api.columns.get_loc('leagueId_df2')
            end_index = rows_only_in_api.columns.get_loc('_merge')
            rows_only_in_api = rows_only_in_api.drop(rows_only_in_api.columns[start_index:end_index + 1], axis=1)
            # Save to db
            rows_only_in_api.to_sql('players_tbl', db_conn, if_exists='append', index=False)
    return


# create_player_list()






def get_list_of_puuid():
    # pull player_list from db ( needs to exist before hand )
    # pull puuid_list from db  ( can create if does not exist)
    # find  all summonerIds that are in player_list but not in puuid_list
    # do an api call for all those summmonerIds to get their respective puuids
    # add these puuids to puuid_db
    # (make sure to rename id to summoner id before saving to db)
    db_conn = connect_to_database(yaml_dict, "league_db_server", "test_league_db")
    query = "SELECT * FROM players_tbl"
    player_list_df = execute_query(db_conn, query)
    query = "select * from puuid_tbl"
    try:
        puuid_df = execute_query(db_conn, query)

    except Exception as e:
        columns = ["id", "accountId", "puuid", "name", "profileIconId", "revisionDate", "summonerLevel"]
        puuid_df = pd.DataFrame(columns=columns)
        print(f"An error occurred: {str(e)}")

    # find all the entries in player_test_table that dont have a corresponding entry in puuid table
    missing_puuid = pd.merge(player_list_df, puuid_df, left_on='summonerId', right_on='id', how='left', indicator=True, suffixes=('', '_df2'))
    missing_puuid = missing_puuid[missing_puuid['_merge'] == 'left_only']
    start_index = missing_puuid.columns.get_loc('id')
    end_index = missing_puuid.columns.get_loc('_merge')
    missing_puuid = missing_puuid.drop(missing_puuid.columns[start_index:end_index + 1], axis=1)
    print(f'Total rows that need to be added ' + str(len(missing_puuid)))
    missing_puuid = missing_puuid.head(10000)
    print(f'Total rows that need to be added ' + str(len(missing_puuid)))
    # When adding 10k rows takes about 12,055 seconds or 200 minutes to complete ~ 50 requests per minute

    # do an api call for each of these summonerIds
    # then append to puuid table in database
    start_time = time.time()
    columns = ["id", "accountId", "puuid", "name", "profileIconId", "revisionDate", "summonerLevel"]
    new_puuids = pd.DataFrame(columns=columns)
    for _index, row in missing_puuid.iterrows():
        # TODO handle failed api call and just continue to next
        api_url = f'https://na1.api.riotgames.com/lol/summoner/v4/summoners/{row["summonerId"]}?api_key={API_KEY}'
        new_id = fetch_api_call(api_url)
        new_id = pd.DataFrame(new_id, index=[0])
        new_puuids = pd.concat([new_puuids, new_id], ignore_index=True)

    new_puuids.to_sql('puuid_tbl', db_conn, if_exists='append', index=False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    return


get_list_of_puuid()

# TODO
# Add 2 column to puuid_tbl, update them while running through create_match_id
#       - 'match_history_processed' default value 0, if they have been through create_match_id_list() then change to 1
#       - 'match_history_process_date' default value null, if they have ever been processed, record the most recent time
#   add this after we get all 25k puuids


def create_match_id_list():
    """
    puuid_tbl (must exist in db)
    match_id (can  create it doesn't exist)

    grab puuid list from db that has not already been processed
    from puuid ask api for 100 match_ids from match history

    check the 100 match ids if they are already in the match_id db tbl
    add new ones to the match_ids tbl
    :return:
    """
    db_conn = connect_to_database(yaml_dict, "league_db_server", "test_league_db")
    query = "SELECT puuid, matches_processed, matches_processed_date FROM puuid_tbl WHERE matches_processed = 0"
    puuid_df = execute_query(db_conn, query)
    query = "SELECT match_id, match_details_processed, match_details_process_date"
    try:
        match_id_df = execute_query(db_conn, query)

    except Exception as e:
        columns = ['matchId', 'matches_processed', 'matches_processed_date']
        match_id_df = pd.DataFrame(columns=columns)
        print(f"An error occurred: {str(e)}")


    for _index, row in puuid_df.iterrows():
        api_url = f'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{row["puuid"]}/ids?queue=420&start=0&count=100&api_key={API_KEY}'
        new_matchs = fetch_api_call(api_url)
        new_matchs = pd.DataFrame({'matchId': new_matchs})

        rows_to_add = df2[~df2['matchId'].isin(df1['matchId'])]
        new_match_id_df = pd.concat([df1, rows_to_add])
        new_match_id_df.reset_index(drop=True, inplace=True)



def get_match_details(match_id, api_key):
    """
    produces the details of a specific match
    :param match_id:
    :param api_key:
    :return: dictionary -> match details
    """











