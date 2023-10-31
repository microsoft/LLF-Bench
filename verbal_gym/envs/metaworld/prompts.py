
mw_instruction = (
    "Your goal is control a Sawyer robot to solve a {task} task. You would see observation of the robot's state and the world state through a json string. Your task is to control the robot to achieve the goal state. Your action is a 4-dim vector, where the first 3 dimension controls the xyz movement of the robot's end effector, and the last dimension controls the gripper (0 means open, and 1 means close.) You action action sets the target goal of the robot. The robot will move to the target goal with a P controller.",
)

r_feedback = (
    "You received reward of {reward}.",
)

fp_feedback = (
    "You should go to  \n{expert_action}.",
)

open_gripper_feedback = (
    "You should open the gripper.",
)

close_gripper_feedback = (
    "You should close the gripper.",
)

hp_feedback = (
    "You're getting closer. You moved in the right direction. Keep going!",
)

hn_feedback = (
    "You moved in the wrong direction. You're now futhre away from the goal than before.",
)
