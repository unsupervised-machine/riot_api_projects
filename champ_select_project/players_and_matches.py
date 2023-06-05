import random
import requests
import yaml
import time
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
    if response.status_code == 429:
        print('Waiting due to rate limit...')
        time.sleep(20)
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

# read the puuid database table if it exists
# convert it to dataframe
# call the new data from riot api
# convert it to dataframe
# merge to find only new puuids
# add these entries to the table

# pull from api

from_api = get_leagues(api_key=API_KEY, region='na1', queue_type='RANKED_SOLO_5x5', tier='DIAMOND', divison='I', page='1')
from_api = pd.DataFrame(from_api)
from_api = from_api.drop('miniSeries', axis=1)
if 'miniSeries' in from_api.columns:
    from_api = from_api.drop('miniSeries', axis=1)

# reads from database table
db_conn = connect_to_database(yaml_dict, "league_db_server", "test_league_db")
query = "SELECT * FROM players_tbl"
from_db = execute_query(db_conn, query)

# use this as an example of things I can do with database
# finds all rows not already in the database
merged = from_api.merge(from_db, on='summonerId', how='left', indicator=True, suffixes=('', '_df2'))
# remove rows that are found in right table
rows_only_in_api = merged[merged['_merge'] == 'left_only']
# remove extra set of columns
start_index = rows_only_in_api.columns.get_loc('leagueId_df2')
end_index = rows_only_in_api.columns.get_loc('_merge')
rows_only_in_api = rows_only_in_api.drop(rows_only_in_api.columns[start_index:end_index+1], axis=1)

# add these new rows to the database table
rows_only_in_api.to_sql('players_tbl', db_conn, if_exists='append', index=False)

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


create_player_list()


def get_matches_from_player_id(api_key, player_id, queue_type):
    """
    produces list of match ids from a players recent matches, can specify queue type(ranked,aram,etc.)
    :param player_id:
    :param api_key:
    :return:list of matches
    """
    pass


def get_match_details(match_id, api_key):
    """
    produces the details of a specific match
    :param match_id:
    :param api_key:
    :return: dictionary -> match details
    """
    pass


def save_dataframe_to_database(dataframe, table_name, database_url):
    """
    saves a dataframe to a postgresql database - duh.
    :param dataframe:
    :param table_name:
    :param database_url:
    :return: none
    """
    # Create a SQLAlchemy engine to connect to the PostgreSQL database
    engine = create_engine(database_url)

    # Convert the DataFrame to a SQL table and save it to the database
    dataframe.to_sql(table_name, engine, if_exists='replace', index=False)

    # Optional: Print a success message
    print("Data saved successfully to the database!")
    return




