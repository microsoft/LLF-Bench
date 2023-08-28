import random
from verbal_gym.utils.misc_utils import extract_int

def extract_action(response, n_actions, separator="#"):
    if n_actions is None:  # free form action
        action = response.split(separator)[1]
        return action
    try:
        action = response.split(separator)[1]
        if not action.isnumeric():
            action = extract_int(action)[0]
        action = int(action)
        if action>=0 and action<n_actions:
            return action
        else:
            print("Action {} is out of range [0, {}].\n".format(action, n_actions-1))
    except IndexError:
        pass
    print("Cannot find the action in the response, so take a random action.\n\tResponse: {}.\n".format(response))
    return random.randint(0,n_actions-1)

if __name__=='__main__':

    response = 'Decision: #0 or #1 (either action can be chosen randomly).'
    print(response, '\n', f'action: {extract_action(response, 2)}\n')
    response = 'Decision: #0#'
    print(response, '\n', f'action: {extract_action(response, 2)}\n')
    response = 'Decision: #Take a random action.#'
    print(response, '\n', f'action: {extract_action(response, 2)}\n')
