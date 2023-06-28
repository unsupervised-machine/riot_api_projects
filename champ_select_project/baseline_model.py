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
from sklearn.metrics import precision_score
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

class_labels = {
    1: "engange_tank",
    2: "adc",
    3: "apc",
    4: "adfighter",
    5: "apfighter",
    6: "enchanter",
    7: "assassins",
    8: "adpoke",
    9: "ap_poke"
}
champion_id_to_class = {
    1: 3, 2: 4, 3: 1, 4: 9, 5: 4, 6: 4, 7: 7, 8: 3, 9: 1, 10: 2, 11: 7, 12: 1, 13: 3, 14: 1, 15: 2, 16: 6, 17: 9,
    18: 2, 19: 4, 20: 1, 21: 2, 22: 8, 23: 4, 24: 4, 25: 9, 26: 6, 27: 1, 28: 7, 29: 2, 30: 3, 31: 1, 32: 1, 33: 1,
    34: 3, 35: 7, 36: 1, 37: 6, 38: 5, 39: 4, 40: 6, 41: 8, 42: 9, 43: 9, 44: 6, 45: 3, 48: 4, 50: 5, 51: 2, 53: 1,
    54: 1, 55: 5, 56: 4, 57: 9, 58: 4, 59: 1, 60: 5, 61: 3, 62: 1, 63: 3, 64: 4, 67: 2, 68: 5, 69: 3, 72: 1, 74: 3,
    75: 4, 76: 9, 77: 4, 78: 1, 79: 1, 80: 4, 81: 8, 82: 5, 83: 4, 84: 5, 85: 1, 86: 4, 89: 1, 90: 3, 91: 7, 92: 4,
    96: 2, 98: 1, 99: 9, 101: 9, 102: 9, 103: 3, 104: 2, 105: 8, 106: 4, 107: 4, 110: 8, 111: 1, 112: 3, 113: 1,
    114: 4, 115: 9, 117: 6, 119: 2, 120: 1, 121: 7, 122: 4, 126: 8, 127:1, 131: 1, 133: 2, 134: 3, 136: 5, 141: 4, 142: 9,
    143: 9, 145: 9, 147: 6, 150: 1, 154: 1, 157: 4, 161: 9, 163: 3, 164: 4, 166: 2, 201: 1, 202: 2, 203: 2, 221: 2,
    222: 2, 223: 1, 234: 4, 235: 2, 236:2, 238: 7, 240: 4, 245: 7, 246: 7, 254: 4, 266:3, 267: 6, 268:3,  350: 6, 360: 2,
    412: 1, 421: 4, 427: 6, 429: 2, 420: 4, 421: 4, 429: 2, 432:5, 497:1, 498:2, 526: 1, 516:1, 518: 3, 555: 7, 517: 5,
    523: 2, 518: 3, 555: 7, 526: 1, 555: 7, 711:3, 777:4, 875: 4, 876: 5, 888: 6, 200:4, 887:5,  895:2, 897:1, 902:6
}

champion_labels = pd.DataFrame(champion_id_to_class.items(), columns=['champion_id', 'label'])
champion_labels['class_label'] = champion_labels['label'].map(class_labels)
df = df.merge(champion_labels, on='champion_id', how='left')

#any unlabeled data
test = df[df['class_label'].isnull()]


df = df.drop(['champion_id', 'champion_name', 'label'], axis=1)



# Let's pivot the data for ML purposes now
df_pivot = pd.pivot_table(df, index=['match_id', 'team_id'], columns='lane_id_1', values='class_label', aggfunc='first')
df_pivot['win'] = df.groupby(['match_id', 'team_id'])['win'].first().values
df_pivot.reset_index(inplace=True)
df_pivot.drop('match_id', axis=1, inplace=True)

df_pivot.reset_index(inplace=True)

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

best_accuracy_value = 0.0
best_accuracy_model = None
best_accuracy_params = None

# best_recall_value = 0.0
# best_recall_model = None
# best_recall_params = None
#
# best_precision_value = 0.0
# best_precision_model = None
# best_precision_params = None


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
        # precision = precision_score(y_test, y_pred)



        print("Model: ", model)
        print("Parameters: ", param_combination)
        print("Accuracy: ", accuracy)
        # print("recall ", recall)
        # print("precision ", precision)
        print("-----------")

        # Update the best model and parameters if the accuracy is higher
        if accuracy > best_accuracy_value:
            best_accuracy_value = accuracy
            best_accuracy_model = model
            best_accuracy_params = param_combination

        # if recall > best_recall_value:
        #     best_recall_value = recall
        #     best_recall_model = model
        #     best_recall_params = param_combination
        #
        # if precision > best_precision_value:
        #     best_precision_value = precision
        #     best_precision_model = model
        #     best_precision_params = param_combination

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

print("Best Model accuracy: ", best_accuracy_model)
print("Best Parameters accuracy: ", best_accuracy_params)
print("Best Accuracy value: ", best_accuracy_value)

# print("Best Model recall: ", best_accuracy_model)
# print("Best Parameters recall: ", best_accuracy_params)
# print("Best Accuracy value: ", best_accuracy_value)
#
# print("Best Model precision: ", best_accuracy_model)
# print("Best Parameters precision: ", best_accuracy_params)
# print("Best Accuracy value: ", best_accuracy_value)



# Save model results to table in db
current_date = datetime.now()
formatted_date = current_date.strftime("%Y%m%d%H%M%S")
date = formatted_date
label = 'baseline_model'
model_name = label + formatted_date
number_of_rows_start = len(df_copy)
metric_validated_against = 'test'
metric_name =
metric_value =
model =

# metric_name = 'accuracy'
# metric_value = 0.5187853834276891
# model = "LogisticRegression(C=10, solver='saga')"
# parameters = "{'C': 10, 'solver': 'liblinear'}"


columns = ['date', 'label', 'model_name', 'number_of_rows_start', 'metric_validated_against', 'metric_name',
           'metric_value', 'model', 'parameters']
data = [[date, label, model_name, number_of_rows_start, metric_validated_against, metric_name,
        metric_value, model, parameters]]

add_to_db = pd.DataFrame(data, columns=columns)


add_to_db.to_sql('model_results', db_conn, if_exists='append', index=False)

