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
print("Number of player_matches: ", len(df))
print("Number of matches: ", len(df['match_id'].unique()))
print("Number of non_matching_lanes: ", len(df[df['lane_id_1'] != df['lane_id_2']]))
print("Number of games_with_non_matching_lanes: ", len(df['match_id'][df['lane_id_1'] != df['lane_id_2']].unique()))
# From our 20k games 672 players did not go to the right lane in 621 games, that is about 3% of the games.
# Feel comfortable dropping them for now.
non_matching_lanes = df[df['lane_id_1'] != df['lane_id_2']]
non_matching_games = non_matching_lanes['match_id'].unique()
df = df[~df['match_id'].isin(non_matching_games)]
print("New number of matches: ", len(df['match_id'].unique()))

# We can drop one of the lane_id columns since they now contain the same info
df = df.drop('lane_id_2', axis=1)

# check for missing data
# 0 missing data, I do not need to deal with this problem yet
missing_data = df[df.isnull().any(axis=1)]
print("Number of rows with any missing data: ", len(missing_data))

# Are the data types appropriate?
# yes
df.dtypes


# Create new columns recording if team1/team2 won or not
conditions = [
    (df['team_id'] == 100) & (df['win'])
]
values = [True]
values_2 = [False]
df['team_1_win'] = np.select(conditions, values, default=False)
df['team_2_win'] = np.select(conditions, values_2, default=True)


# Let's pivot the data for ML purposes now
df_pivot = df.pivot(index='match_id', columns=['team_id', 'lane_id_1'], values=['champion_name'])
df_pivot['team_1_win'] = df.groupby('match_id')['team_1_win'].first()
df_pivot['team_2_win'] = df.groupby('match_id')['team_2_win'].first()
# rename the pivoted columns to be descriptive
new_column_names = ['team_1_top', 'team_1_jng', 'team_1_mid', 'team_1_adc', 'team_1_sup',
                    'team_2_top', 'team_2_jng', 'team_2_mid', 'team_2_adc', 'team_2_sup',
                    'team_1_win', 'team_2_win']
df_pivot.columns = new_column_names
df = df_pivot


# Since the champion columns are not numeric we should one hot encode them
columns_to_encode = df.columns[0:10]
df_encoded = pd.get_dummies(df, columns=columns_to_encode)
bool_columns = df_encoded.select_dtypes(include=bool).columns
df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)
df = df_encoded





# implement variance threshold feature selection to deal with the sparse data
selector = VarianceThreshold(threshold=0.01)
selected_features = selector.fit_transform(df)

selected_feature_names = df.columns[selector.get_support()]

# went from 1300 features to 315
selected_df = pd.DataFrame(selected_features, columns=selected_feature_names)

df = selected_df

# Calculate the total number of cells in the DataFrame
total_cells = df.size

# Calculate the number of missing or null values in the DataFrame
missing_cells = (df == 0).sum().sum()

# Calculate the sparsity percentage
sparsity = (missing_cells / total_cells) * 100

# Print the sparsity percentage
# still too sparse
print(f"Sparsity: {sparsity}%")







# implement embedding













# super sparse matrix
    # sk learn tools

# signal to noise ratio


# neural network
    # embeddings
    # optimal embedding depth




# remove intermediate dataframes and objs from memory
del df_pivot, missing_data, non_matching_lanes, df_encoded
del conditions, values, values_2, new_column_names, non_matching_games, bool_columns, columns_to_encode
gc.collect()



# Let's start doing some ML stuff
# Create test and train set ( also create holdout set )
X = df.drop(['team_1_win', 'team_2_win'], axis=1)
y = df['team_1_win']
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

