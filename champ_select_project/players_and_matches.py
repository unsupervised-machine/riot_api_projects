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
    :param data_base: string, target database to connect to
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


# TODO write another file that explains how to use function with examples
# Example how to pull data from database
test_connection = connect_to_database(yaml_dict, 'league_db_server', 'test_league_db')
query = "SELECT * FROM sample_table"
test = execute_query(test_connection, query)





def get_leagues(api_key, region):
    """
    produces a list of leagues (types of ranked ladders) for a given region
    :param region:
    :return: list of league_ids
    """
    pass


def get_players_from_league(api_key, league_id):
    """
    produces list of player ids from a given league
    :param league_id:
    :param api_key:
    :return: list of player ids
    """
    pass


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


