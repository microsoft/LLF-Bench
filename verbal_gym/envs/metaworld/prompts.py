
mw_instruction = (
    "Your goal is control a Sawyer robot to solve a {task} task. You would see observation of the robot's state and the world state through a json string. Your task is to control the robot to achieve the goal state. Your action is a 4-dim vector, where the first 3 dimension controls the xyz movement of the robot's end effector, and the last dimension controls the gripper (0 means open, and 1 means close.)",
)

r_feedback = (
    "You received reward of {reward}.",
)

fp_feedback = (
    "You should have taken action\n{expert_action}.",
)