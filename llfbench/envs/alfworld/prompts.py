reward_descp = [
    "You got a reward of {reward}.",
    "You receive a reward of {reward}.",
    "The reward you get for this action is {reward}.",
    "Your latest action gives you a reward of {reward}.",
    "The reward you are owed for your last action is {reward}."
]

# HN
mistake_bad_action_descp = [
    "You made a mistake by taking the bad action {avoid_action}.",
    "You should not have taken the bad action {avoid_action}.",
    "In your last move, you should have avoided the bad action {avoid_action}.",
    "It was a wrong decision to take the action {avoid_action}.",
    "It was a wrong decision to take {avoid_action}.",
]

correct_bad_action_descp = [
    "You were correct in not taking the bad action {avoid_action}.",
    "You did a good job in not taking the bad action {avoid_action}.",
    "It was a right decision to not take the bad action {avoid_action}.",
    "At the last step, you did not take the action {avoid_action}, and this was a good thing as it was a bad action.",
    "Not taking the action {avoid_action} was a good thing.",
]

# HP
mistake_good_action_descp = [
    "You should have taken the {past_opt_action} action.",
    "In the last step, you should have taken the {past_opt_action} action.",
    "You made a mistake by not taking the {past_opt_action} action.",
    "The action {past_opt_action}, is what you should have chosen in your last move.",
    "You should have taken the action {past_opt_action} instead.",
]

correct_good_action_descp = [
    "You did the right thing by taking the {past_opt_action} action.",
    "You were right to take the {past_opt_action} action.",
    "Good job at taking the correct action {past_opt_action} in the last move.",
    "You took the right decision to follow {past_opt_action}.",
    "Following {past_opt_action} was the right thing to do."
]

# FP
follow_opt_action_descp = [
    "You should now take the {opt_action} action.",
    "In the next step, follow the good action {opt_action}.",
    "You must follow the good action {opt_action} in this coming step.",
    "The optimal action to take in the next step is {opt_action}.",
    "Take {opt_action} in the next step."
]

# FN
avoid_bad_action_descp = [
    "You should not take the action {avoid_action} in the next step.",
    "You should avoid the action {avoid_action} in the next step.",
    "Avoid the {avoid_action} action, in your next move.",
    "Do not take the {avoid_action} action in your next move.",
    "In the next step, do not take the {avoid_action} action.",
]
