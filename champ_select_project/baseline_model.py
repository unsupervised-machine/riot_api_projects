from sqlalchemy import create_engine, text
import yaml
import gc


import time
import random
from datetime import datetime


import pandas as pd
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import OneHotEncoder




from sklearn.feature_selection import VarianceThreshold


# need to install these (wait til other scripts are done running)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam


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



# Start Script

# Get champ select data
db_conn = connect_to_database(yaml_dict, "league_db_server", "test_league_db")
query = "select * from public.champion_select_details_tbl"
df = execute_query(db_conn, query)
df_copy = df.copy()
df = df_copy.copy()
# we currently have 200510 rows, we might want to add row index to this data in the future....


# deal with games where people did not go to assigned lanes
# print("Number of player_matches: ", len(df))
# print("Number of matches: ", len(df['match_id'].unique()))
# print("Number of non_matching_lanes: ", len(df[df['lane_id_1'] != df['lane_id_2']]))
# print("Number of games_with_non_matching_lanes: ", len(df['match_id'][df['lane_id_1'] != df['lane_id_2']].unique()))
# From our 20k games 672 players did not go to the right lane in 621 games, that is about 3% of the games.
# Feel comfortable dropping them for now.
non_matching_lanes = df[df['lane_id_1'] != df['lane_id_2']]
non_matching_games = non_matching_lanes['match_id'].unique()
df = df[~df['match_id'].isin(non_matching_games)]
# print("New number of matches: ", len(df['match_id'].unique()))

# We can drop one of the lane_id columns since they now contain the same info
df = df.drop('lane_id_2', axis=1)

# check for missing data
# 0 missing data, I do not need to deal with this problem yet
# missing_data = df[df.isnull().any(axis=1)]
# print("Number of rows with any missing data: ", len(missing_data))


# there are Latin america games:
# list how many, remove them

# Are the data types appropriate?
# yes
# df.dtypes


# Let's pivot the data for ML purposes now
df_pivot = pd.pivot_table(df, index=['match_id', 'team_id'], columns='lane_id_1', values='champion_name', aggfunc='first')
df_pivot['win'] = df.groupby(['match_id', 'team_id'])['win'].first().values
df_pivot.reset_index(inplace=True)
df_pivot.drop('match_id', axis=1, inplace=True)

# Do we have too many categories?
# Distinct values for each column
temp_df = df_pivot.copy()
temp_df.drop(['team_id', 'win'], axis=1,inplace=True)
distinct_df_before = temp_df.apply(lambda x: pd.Series(x.unique()))
# Transpose the distinct DataFrame
distinct_df_before = distinct_df_before.T

# Let's only keep teams with the top 100 most played champions in each position
threshold = len(df_pivot) * 0.01  # Calculate the threshold for 1%
rows_to_drop = set()  # Set to collect row indices to be dropped
for column in df_pivot.columns:
    value_counts = df_pivot[column].value_counts()  # Count occurrences of each value
    infrequent_values = value_counts[value_counts < threshold].index  # Get values occurring less than threshold
    # Collect row indices to be dropped
    rows_to_drop.update(df_pivot[df_pivot[column].isin(infrequent_values)].index)

# what percent of rows are we about to drop
# went from 140+ categories in each role to between 15-36 for each role
len(rows_to_drop) / len(df_pivot) # 48% thats okay I guess? Can look into this effect later

# Drop all collected rows at once
df_pivot = df_pivot.drop(rows_to_drop)
df_pivot.reset_index(inplace=True)

# Distinct values for each column
temp_df = df_pivot.copy()
temp_df.drop(['index', 'team_id', 'win'], axis=1,inplace=True)
distinct_df_after = temp_df.apply(lambda x: pd.Series(x.unique()))
# Transpose the distinct DataFrame
distinct_df_after = distinct_df_after.T
df = df_pivot.copy()

df.drop(['index', 'team_id'], axis=1, inplace=True)
# need to encode the categories so that a ML model can use them
# Perform one-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(df.iloc[:, :-1])

# Create a new DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(df.columns[:-1].tolist()))

# Combine the encoded features with the 'Result' column
final_df = pd.concat([encoded_df, df['win']], axis=1)

# Display the final encoded DataFrame
print(final_df)

df = final_df.copy()


# Let's start doing some ML stuff
# Create test and train set ( also create holdout set )
X = df.drop(['win'], axis=1)
y = df['win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# Try various plausible models and hyperparameters and find best one
models = [
    {'model': LogisticRegression(), 'params': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}},
    {'model': DecisionTreeClassifier(), 'params': {'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}},
    {'model': RandomForestClassifier(), 'params': {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}},
    {'model': GradientBoostingClassifier(), 'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 1.0]}},
    {'model': xgb.XGBClassifier(), 'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 1.0]}},
    # {'model': Sequential(), 'params': {'hidden_layers': [(64,), (128,), (64, 32)], 'activation': ['relu', 'sigmoid'], 'lr': [0.001, 0.01, 0.1]}}
]

best_accuracy = 0.0
best_recall = 0.0
best_model = None
best_params = None

# Loop over each model and hyperparameter combination
start_time = time.time()
for model_info in models:
    model = model_info['model']
    params = model_info['params']

    # Loop over each combination of hyperparameters
    for param_combination in ParameterGrid(params):
        model.set_params(**param_combination)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        # recall = recall_score(y_test, y_pred)


        print("Model: ", model)
        print("Parameters: ", param_combination)
        print("Accuracy: ", accuracy)
        print("-----------")

        # Update the best model and parameters if the accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_params = param_combination


end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

print("Best Model: ", best_model)
print("Best Parameters: ", best_params)
print("Best Accuracy: ", best_accuracy)


# Save model results to table in db
current_date = datetime.now()
formatted_date = current_date.strftime("%Y%m%d%H%M%S")
date = formatted_date
label = 'baseline_model'
model_name = label + formatted_date
number_of_rows_start = len(df_copy)
metric_validated_against = 'test'
metric_name = 'accuracy'
metric_value = 0.5187853834276891
model = "LogisticRegression(C=10, solver='saga')"
parameters = "{'C': 10, 'solver': 'liblinear'}"


columns = ['date', 'label', 'model_name', 'number_of_rows_start', 'metric_validated_against', 'metric_name',
           'metric_value', 'model', 'parameters']
data = [[date, label, model_name, number_of_rows_start, metric_validated_against, metric_name,
        metric_value, model, parameters]]

add_to_db = pd.DataFrame(data, columns=columns)


add_to_db.to_sql('model_results', db_conn, if_exists='append', index=False)

