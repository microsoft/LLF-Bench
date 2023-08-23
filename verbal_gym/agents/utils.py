import random
from verbal_gym.utils.misc_utils import extract_int

def extract_action(response, n_actions, separator="#"):
    if n_actions is None:  # free form action
        action = response.split(separator)[1]
        return action
    try:
        action = response.split(separator)[1]
        if action.isnumeric():
            action = int(action)
            if action>=0 and action<n_actions:
                return action
    except IndexError:
        pass
    print("Cannot find the action in the response, so take a random action.\n\tResponse: {}.\n".format(response))
    return random.randint(0,n_actions-1)