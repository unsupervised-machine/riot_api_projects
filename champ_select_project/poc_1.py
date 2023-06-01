import requests
import yaml
import time
# Proof of concept for pulling match data using player uid

yaml_file = 'champ_select_project/secrets.yml'
with open(yaml_file, 'r') as file:
    keys = yaml.safe_load(file)
API_KEY = keys['app_key']


def check_params(*args):
    for arg in args:
        if arg is None:
            raise ValueError("None value detected in function parameters.")



def get_player_uid(api_key=None, player_name=None):
    if player_name is None or api_key is None:
        raise ValueError("Inappropriate Parameters")
    api_url = 'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/' + player_name + '?api_key=' + api_key
    resp = requests.get(api_url)
    player_info = resp.json()
    player_uid = player_info['puuid']
    return player_uid


def get_most_recent_match(api_key=None, player_uid=None):
    if api_key is None or player_name is None:
        raise ValueError("Inappropriate Parameters")
    api_url = 'https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/' + player_uid + '/ids?start=0&count=20' + '&api_key=' + api_key
    resp = requests.get(api_url)
    recent_matches = resp.json()
    most_recent_match = recent_matches[0]
    return most_recent_match


def get_match_data(api_key=None, region=None, match_id=None):
    if api_key is None or region is None or match_id is None:
        raise ValueError("Inappropriate Parameters")
    api_url = (
            'https://' + region + '.api.riotgames.com/lol/match/v5/matches/' + match_id + '?api_key=' + api_key
    )
    response = requests.get(api_url)
    match_data = response.json()
    return match_data


def did_win(api_key=None, match_data=None, player_uid=None):
    if api_key is None or match_data is None or player_uid is None:
        raise ValueError("Inappropriate Parameters")
    player_match_participant_index = match_data['metadata']['participants'].index(player_uid)
    player_match_data = match_data['info']['participants'][player_match_participant_index]
    player_won = player_match_data['win']
    return player_won


def get_match_ids(region=None, player_uid=None, count=5, api_key=None):
    """
    :param region:
    :param player_uid:
    :param count:
    :param api_key:
    :return: a list of match_ids
    """
    api_url = (
            'https://' +
            region +
            '.api.riotgames.com/lol/match/v5/matches/by-puuid/' +
            player_uid +
            '/ids' +
            '?type=ranked&' +
            'start=0&' +
            'count=' +
            str(count) +
            '&api_key=' +
            api_key
    )

    while True:
        response = requests.get(api_url)

        if response.status_code == 429:
            print('waiting')
            time.sleep(20)
            continue
        data = response.json()
        return data


# def test_get_match_ids():
#     api_key = API_KEY
#     region = 'americas'
#     player_uid = get_player_uid(API_KEY, 'taran')
#     count = 5
#     matches = get_match_ids(region, player_uid, count, api_key)
#     return matches
#
# example_match_ids = test_get_match_ids()


def get_match_data(region='americas', count=5, api_key=API_KEY, player_list = []):
    """
    Retrieve match data for a player from the Riot Games API.

    Args:
        region (str, optional): The region where the player is located. Defaults to 'americas'.
        count (int, optional): The number of recent matches to retrieve. Defaults to 5.
        api_key (str, optional): The API key for accessing the Riot Games API. Defaults to API_KEY.
        player_name (str, optional): The name of the player. Defaults to 'taran'.

    Returns:
        dict: A dictionary containing match data for each match played by the player.
            The keys are match IDs, and the values are JSON objects containing detailed match information.
    """

    if not player_list:
        # the list is empty
        player_list.append(get_player_uid(api_key, 'taran'))


    # Check the validity of the parameters
    check_params(region, count, api_key, player_list)

    match_list = []
    for player_uid in player_list:
        # Retrieve the match IDs for the player in the specified region
        new_match = get_match_ids(region, player_uid, count, api_key)
        match_list.extend(new_match)

    # Remove duplicate matches from the list
    match_list = list(set(match_list))

    # Fetch detailed match data for each match ID
    match_dict = {}
    for match_id in match_list:
        api_url = (
                'https://' +
                region +
                '.api.riotgames.com/lol/match/v5/matches/' +
                match_id +
                '?api_key=' +
                api_key
        )
        response = requests.get(api_url)

        # Handle rate limit (429) errors by waiting and retrying after a delay
        if response.status_code == 429:
            print('Waiting due to rate limit...')
            time.sleep(20)
            continue

        # Parse the JSON response into match data and store it in the dictionary
        match_data = response.json()
        match_dict[match_id] = match_data

    return match_dict


# def test_get_match_data():
#     match_data = get_match_data(count=5)
#     return match_data


# example_match_data = test_get_match_data()
# Check that all the games are ranked solo (queueId == 420)
# TODO I am including a few non ranked games remember to filter them out eventually
# for match_id, inner_dict in example_match_data.items():
#     queueType = inner_dict['info']['queueId']
#     if queueType != 420:
#         raise ValueError("Including wrong queue type")

# TODO generate a list of players to test functionality of get_match_data()


