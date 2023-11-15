
mw_instruction = (
    "Your goal is control a Sawyer robot to solve a {task} task. You would see observation of the robot's state and the world state through a json string. Your task is to control the robot to achieve the goal state. Your action is a 4-dim vector, where the first 3 dimension controls the xyz movement of the robot's end effector, and the last dimension controls the gripper (0 means open, and 1 means close.) You action action sets the target goal of the robot. The robot will move to the target goal with a P controller.",
)

r_feedback = (
    "Your reward for the latest step is {reward}.",
    "You got a reward of {reward}.",
    "The latest step brought you {reward} reward units.",
    "You've received a reward of {reward}.",
    "You've earned a reward of {reward}.",
    "You just got {reward} points.",
    "{reward} points for you.",
    "You've got yourself {reward} units of reward.",
    "The reward your latest step earned you is {reward}.",
    "The previous step's reward was {reward}.",
    "+{reward} reward",
    "Your reward is {reward}.",
    "The reward you just earned is {reward}.",
    "You have received {reward} points of reward.",
    "Your reward={reward}.",
    "The reward is {reward}.",
    "Alright, you just earned {reward} reward units.",
    "Your instantaneous reward is {reward}.",
    "Your rew. is {reward}.",
    "+{reward} points",
    "Your reward gain is {reward}."
)

fp_feedback = (
    "You should go to  \n{expert_action}.",
    "It's recommended that you proceed to \n{expert_action}.",
    "You are advised to head to \n{expert_action}.",
    "It would be best if you go to \n{expert_action}.",
    "You ought to go to \n{expert_action}.",
    "It's suggested that you navigate to \n{expert_action}.",
    "You're encouraged to move to \n{expert_action}.",
    "It's advisable for you to go to \n{expert_action}.",
    "You're recommended to proceed to \n{expert_action}.",
    "It's preferable for you to go to \n{expert_action}.",
    "You're urged to head to \n{expert_action}.",
    "It's proposed that you go to \n{expert_action}.",
    "You're counseled to navigate to \n{expert_action}.",
    "It's advised that you move to \n{expert_action}.",
    "You're suggested to go to \n{expert_action}.",
    "It's recommended for you to proceed to \n{expert_action}.",
    "You're advised to go to \n{expert_action}.",
    "It's suggested for you to head to \n{expert_action}.",
    "You're recommended to navigate to \n{expert_action}.",
    "It's advisable that you move to \n{expert_action}.",
    "You're urged to go to \n{expert_action}.",
)

open_gripper_feedback = (
    "You should open the gripper.",
    "The gripper needs to be opened.",
    "It's necessary to open the gripper.",
    "You are required to open the gripper.",
    "You need to open the gripper.",
    "You ought to open the gripper.",
    "You must open the gripper.",
    "You have to open the gripper.",
    "It's essential to open the gripper.",
    "You're supposed to open the gripper.",
     "You're expected to open the gripper.",
     "You're required to open the gripper.",
     "You're obliged to open the gripper.",
     "You're meant to open the gripper.",
     "You're to open the gripper.",
     "You're advised to open the gripper.",
     "You're recommended to open the gripper.",
     "You're suggested to open the gripper.",
     "You're instructed to open the gripper.",
     "You're directed to open the gripper.",
     "You're ordered to open the gripper.",
)

close_gripper_feedback = (
    "You should close the gripper.",
    "You need to shut the gripper.",
    "It's necessary to close the gripper.",
    "You are required to close the gripper.",
    "You ought to close the gripper.",
    "You must close the gripper.",
    "You have to close the gripper.",
    "It's essential to close the gripper.",
    "You're supposed to close the gripper.",
    "You're expected to close the gripper.",
     "You're obliged to close the gripper.",
     "You're required to shut the gripper.",
     "You're expected to shut the gripper.",
     "You're obliged to shut the gripper.",
     "You're supposed to shut the gripper.",
     "You need to seal the gripper.",
     "You ought to seal the gripper.",
     "You must seal the gripper.",
     "You have to seal the gripper.",
     "It's necessary to seal the gripper.",
     "It's essential to seal the gripper.",
)

hp_feedback = (
    "You're getting closer. You moved in the right direction. Keep going!",
    "You're making progress. You're heading the right way. Don't stop!",
    "You're nearing your goal. You're on the right path. Continue!",
    "You're approaching your target. You're moving correctly. Keep it up!",
    "You're almost there. You're going the right way. Keep moving!",
    "You're getting nearer. You're moving correctly. Don't give up!",
    "You're drawing closer. You're heading in the right direction. Keep pushing!",
    "You're closing in. You're on the correct path. Keep advancing!",
    "You're nearing your destination. You're moving the right way. Keep proceeding!",
    "You're getting closer to your goal. You're heading the right way. Keep on going!",
     "You're almost at your target. You're moving in the correct direction. Keep progressing!",
     "You're getting nearer to your aim. You're going the right way. Keep moving forward!",
     "You're drawing nearer to your goal. You're heading correctly. Keep on moving!",
     "You're closing in on your target. You're on the right track. Keep going forward!",
     "You're nearing your objective. You're moving the right way. Keep on advancing!",
     "You're getting closer to your destination. You're heading in the correct direction. Keep pushing forward!",
     "You're almost at your aim. You're moving correctly. Keep progressing forward!",
     "You're getting nearer to your target. You're going the right direction. Keep on proceeding!",
     "You're drawing nearer to your objective. You're heading the right way. Keep moving ahead!",
     "You're closing in on your goal. You're on the correct track. Keep going ahead!",
     "You're nearing your aim. You're moving in the right direction. Keep on pushing!",
)

hn_feedback = (
    "You moved in the wrong direction. You're now further away from the goal than before.",
    "You've gone in the incorrect direction. You're currently more distant from the target than previously."
    "You've shifted in the inappropriate path. You're now farther from the objective than earlier."
    "You've proceeded in the wrong way. You're presently further from the aim than before."
    "You've traveled in the incorrect route. You're now more remote from the goal than previously."
    "You've wandered in the wrong direction. You're currently farther from the endpoint than earlier."
    "You've strayed in the inappropriate direction. You're now more distanced from the target than before."
    "You've veered in the wrong path. You're presently further from the objective than previously."
    "You've drifted in the incorrect direction. You're now farther from the aim than earlier."
    "You've deviated in the wrong way. You're currently more removed from the goal than before."
     "You've turned in the inappropriate direction. You're now further from the target than previously."
     "You've headed in the wrong path. You're presently farther from the objective than earlier."
     "You've steered in the incorrect direction. You're now more separated from the aim than before."
     "You've navigated in the wrong way. You're currently further from the goal than previously."
     "You've journeyed in the inappropriate direction. You're now farther from the target than earlier."
     "You've treaded in the wrong path. You're presently more distant from the objective than before."
     "You've meandered in the incorrect direction. You're now further from the aim than previously."
     "You've roamed in the wrong way. You're currently farther from the goal than earlier."
     "You've ambled in the inappropriate direction. You're now more far-off from the target than before."
     "You've sauntered in the wrong path. You're presently further from the objective than previously."
     "You've perambulated in the incorrect direction. You're now farther from the aim than earlier."
)
