import pdb
import gym
import sys
import random

from collections import deque
from verbal_gym.envs.gridworld.room import Room
from verbal_gym.envs.gridworld.scene import Scene


class Gridworld(gym.Env):

    # Basic (b), partial (p), and complete (c)
    INSTRUCTION_TYPES = ('b', 'p', 'c')

    # Feedback type:
    # n: none
    # m: mixed
    # r: reward
    # hn: hindsight negative
    # hp: hindsight positive
    # fn: future negative
    # fp: future positive
    FEEDBACK_TYPES = ('n', 'm', 'r', 'hn', 'hp', 'fn', 'fp')

    # feedback_level="gold"
    def __init__(self, num_rooms=20, horizon=20, fixed=True, instruction_type="c", feedback_type="hp", min_goal_dist=4):
        super(Gridworld, self).__init__()

        # Action space consists of 4 actions: North, South, East and West
        self.num_actions = 4
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Text(sys.maxsize)

        self.instruction_type = instruction_type
        self.feedback_type = feedback_type

        self.num_rooms = num_rooms
        self.horizon = horizon
        assert self.horizon >= 5, "Horizon must be at least 5 to allow agent to somewhat explore the world"
        self.min_goal_dist = min_goal_dist

        self.fixed = fixed

        # Docstring
        self.base_docstring = "You are in a house with multiple rooms. Each room can have objects that will " \
                              "be visible to you if you are in that room. Each room can have a door along the " \
                              "North, South, East and West direction. You can follow a direction to go from one room " \
                              "to another. If there is no door along that direction, then you will remain in the " \
                              "room. You will start in a room. Your goal is to navigate to another room which has " \
                              "the treasure. You have an action space of size 4. Action 0 leads to going North. " \
                              "Action 1 leads to going East. Action 2 leads going west. Action 3 leads to going South."

        # Counters that may have to be reset
        self.docstring = None
        self.current_timestep = 0.0
        self.current_scene = None
        self.current_room = None
        self.goal_prev_visited = False

    def make_scene(self):

        # Process of creating a scene works as follows
        # We start by creating a room
        # We add between 1-4 edges, we create new rooms and add them to the queue

        scene = Scene()
        queue = deque()

        room = scene.create_random_empty_room(pos=(0, 0))
        queue.append(room)

        available_objects = list(Room.OBJECTS)

        while len(queue) > 0 and len(scene.rooms) < self.num_rooms:

            room = queue.popleft()

            # Sample a subset of edges from the list of available directions
            available_directions = [direction for direction in Scene.DIRECTIONS
                                    if direction not in scene.doors[room]]

            if len(available_directions) == 0:
                # All directions from this room has been connected
                continue

            num_dir = random.randint(1, len(available_directions) - 1)
            chosen_directions = random.sample(available_directions, k=num_dir)

            for i, dir_to in enumerate(chosen_directions):

                # TODO Connect the new room not just to where it spawned from but also other rooms to create a graph
                new_pos = scene.get_relative_pos(room, dir_to, length=1)
                ngbr_room = scene.create_random_empty_room(pos=new_pos)
                scene.add_door(room=room,
                               dir_to=dir_to,
                               other_room=ngbr_room)
                queue.append(ngbr_room)

                if len(scene.rooms) >= self.num_rooms:
                    break

        indices = list(range(0, scene.num_rooms()))
        random.shuffle(indices)

        for i in indices:
            if len(available_objects) > 0 and random.random() < 0.25:
                obj = random.choice(available_objects)
                room = scene.get_room(i)
                room.add_object(obj=obj)
                available_objects.remove(obj)

        # Add start room
        rooms = scene.get_rooms()

        goal_room = random.choice(rooms)
        goal_room.add_goal()
        scene.get_add_goal_room(goal_room=goal_room)

        # Do DFS
        scene.start_bfs(goal_room)

        # Add key in a room at least k steps away
        rooms = [ngbr_room for ngbr_room, path in scene.bfs_path.items()
                 if self.min_goal_dist < len(path) < self.horizon - 5 and ngbr_room != goal_room]

        if len(rooms) == 0:
            rooms = [ngbr_room for ngbr_room, path in scene.bfs_path.items() if ngbr_room != goal_room]

        start_room = random.choice(rooms)
        scene.get_add_start_room(start_room=start_room)

        return scene

    def make_room_obs(self, room):
        obs = room.describe_room() + self.current_scene.get_room_doors_description(room)
        return obs

    def reset(self):

        # Counters that may have to be reset
        self.current_timestep = 0.0

        self.current_scene = self.make_scene()

        self.current_room = self.current_scene.get_start_room()
        self.goal_prev_visited = False

        obs = self.make_room_obs(self.current_room)
        self.docstring = self.generate_docstring()

        return dict(instruction=self.docstring,
                    observation=obs,
                    feedback=None)

    def generate_docstring(self):

        if self.instruction_type == "b":
            # Basic docstring
            docstring = self.base_docstring

        elif self.instruction_type == "p":
            # Partial docstring.
            docstring = self.base_docstring + " " + self.get_optimal_path_desc(partial=True)

        elif self.instruction_type == "c":
            # Complete docstring. Optimal policy can be achieved using just the docstring
            docstring = self.base_docstring + " " + self.get_optimal_path_desc(partial=False)
        else:
            raise AssertionError(f"Unhandled feedback_level {self.instruction_type}")

        return docstring

    def get_optimal_path_desc(self, partial=False):

        optimal_path = self.current_scene.get_optimal_path()

        if len(optimal_path) == 0:
            path_descps = ["Congratulations! You are already in the room with treasure."]
        else:
            path_descps = ["I will also provide you a path towards reaching your goal."]

        for ix, (direction, room) in enumerate(optimal_path):
            if ix == 0:
                path_descps.append(f"First, you follow {direction} direction, to reach the room {room.get_name()}.")
            elif ix == len(optimal_path) - 1:
                path_descps.append(f"Next, you follow {direction} direction, to reach the room {room.get_name()}.")
            else:
                path_descps.append(f"Finally, you follow {direction} direction, to reach the "
                                   f"room {room.get_name()} which has treasure.")

        if partial:
            r = 0.4 + random.random() * 0.2
            partial_len = int(len(path_descps) * r)
            opt_path_desc = " ".join(path_descps[:partial_len])
        else:
            opt_path_desc = " ".join(path_descps)
        return opt_path_desc

    def log_env(self, logger):
        self.current_scene.log_scene(logger)

    def step(self, action):

        if self.current_timestep > self.horizon:
            raise AssertionError("Horizon exhausted.")

        old_gold_action = self.current_scene.get_gold_action(self.current_room)

        if 0 <= action < 4:
            new_room = self.current_scene.check_room_door(self.current_room, Scene.DIRECTIONS[action])
        else:
            raise AssertionError(f"Action must be in {{0, 1, 2, 3}} but found {action}")

        if new_room is not None:
            self.current_room = new_room
            next_obs = self.make_room_obs(self.current_room)
        else:
            next_obs = f"You remained in room {self.current_room.get_name()} " \
                       f"as there is no door in the direction {Scene.DIRECTIONS[action]}."

        # Compute the reward
        reward = 1.0 if self.current_room == self.current_scene.goal_room and not self.goal_prev_visited else 0.0
        self.goal_prev_visited = self.goal_prev_visited or (self.current_room == self.current_scene.goal_room)

        # Update the counter and compute done
        self.current_timestep += 1
        done = self.current_timestep == self.horizon

        feedback = self.generate_feedback(self, old_gold_action, new_room)

        info = {
            "feedback": feedback
        }

        next_packed_obs = dict(instruction=self.docstring,
                               observation=next_obs,
                               feedback=feedback)

        return next_packed_obs, reward, done, info

    def generate_feedback(self, old_gold_action, new_room, feedback_type=None):

        if feedback_type is None:
            feedback_type = self.feedback_type

        if feedback_type == "n":        # None
            feedback = "No feedback is available."

        elif feedback_type == "r":      # Reward described in text
            feedback = f"You got a reward of {reward}."
            # if reward == 1.0:
            #     feedback = f"You got a reward of 1.0"
            # elif self.goal_prev_visited:
            #     feedback = f"You have already reached the treasure. Good job."
            # else:
            #     feedback = f"You didn't succeed. Trying visiting more rooms."

        elif feedback_type == "m":      # Mixed feedback type

            sampled_feedback_type = random.choice(["n", "r", "hp", "hn", "fn", "fp"])
            feedback = self.generate_feedback(old_gold_action=old_gold_action,
                                              new_room=new_room,
                                              feedback_type=sampled_feedback_type)

        elif feedback_type == "hn":     # Hindsight negative
            raise NotImplementedError()

        elif feedback_type == "hp":     # Hindsight positive

            if reward == 1.0:
                feedback = f"You succeeded! Congratulations."

            elif self.goal_prev_visited:
                feedback = f"You have already reached the treasure. Good job."

            else:
                # This implies we have not reached the goal neither before or nor in this stage
                # Provide feedback as follows:
                #       Case 1: If the agent takes an action that resulted in no transition
                #       Case 2: If the agent takes the wrong transition
                #       Case 3: If the agent takes the right transition

                if old_gold_action is None:
                    pdb.set_trace()
                assert old_gold_action is not None, "can only be none if you reach the goal"

                if new_room is None:
                    # Case 1: If the agent takes an action that resulted in no transition
                    feedback = f"You tried to go in the direction {Scene.DIRECTIONS[action]} where there is no room. " \
                               f"You should have taken the {old_gold_action} direction."

                elif old_gold_action != Scene.DIRECTIONS[action]:
                    # Case 2: If the agent takes the wrong transition
                    feedback = f"You went through a door to a new room but this is not the direction of the treasure." \
                           f"You should have taken the {old_gold_action} direction."

                else:
                    # Case 3: If the agent takes the right transition
                    feedback = f"Good job. You took the right direction and moved closer to the treasure."

        elif feedback_type == "fn":     # Future negative
            pass

        elif feedback_type == "fp":     # Future positive

            if reward == 1.0:
                feedback = f"You succeeded! Congratulations."

            elif self.goal_prev_visited:
                feedback = f"You have already reached the treasure. Good job."

            else:
                # This implies we have not reached the goal neither before or nor in this stage
                # Provide feedback as follows:
                #       Case 1: If the agent takes an action that resulted in no transition
                #       Case 2: If the agent takes the wrong transition
                #       Case 3: If the agent takes the right transition

                gold_action = self.current_scene.get_gold_action(self.current_room)

                if gold_action is None or old_gold_action is None:
                    pdb.set_trace()

                assert old_gold_action is not None, "can only be none if you reach the goal"
                assert gold_action is not None, "can only be none if you reach the goal"

                if new_room is None:
                    # Case 1: If the agent takes an action that resulted in no transition
                    feedback = f"You tried to go in the direction {Scene.DIRECTIONS[action]} where there is no room. " \
                               f"You should have moved towards the {gold_action} direction."

                elif old_gold_action != Scene.DIRECTIONS[action]:
                    # Case 2: If the agent takes the wrong transition
                    feedback = f"You went through a door to a new room but this is not the direction of the treasure." \
                               f"You should now move towards the {gold_action} direction."

                else:
                    # Case 3: If the agent takes the right transition
                    feedback = f"Good job. You took the right direction and moved closer to the treasure. From this " \
                               f"room you should move towards the {gold_action} direction."

        else:
            raise AssertionError(f"Unhandled feedback level {feedback_type}")

        return feedback


if __name__ == "__main__":
    env = Gridworld(num_rooms=10)
    obs = env.reset()
    print(obs)
    # pdb.set_trace()

    while True:
        action = input("Action is ")
        action = int(action)
        new_obs, reward, done, info = env.step(action)
        print(f"New observation {new_obs}\n, Took action {action}\n, got reward {reward}\n, done {done}\n "
              f"You got feedback is: \n {info['feedback']}.")
        pdb.set_trace()
