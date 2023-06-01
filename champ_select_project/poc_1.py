import requests
import yaml

yaml_file = 'champ_select_project/secrets.yml'
with open(yaml_file, 'r') as file:
    keys = yaml.safe_load(file)
API_KEY = keys['app_key']


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


def get_ranked_matches(region=None, player_uid=None, count=None, api_key=None):
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
            time.sleep('20')
            continue
        data = response.json()
        return data


def test_get_matches():
    api_key = API_KEY
    region = 'americas'
    player_uid = get_player_uid(API_KEY, 'taran')
    count = 5
    matches = get_ranked_matches(region, player_uid, count, api_key)
    return matches

match_ids = test_get_matches()