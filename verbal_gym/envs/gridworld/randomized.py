import pdb
import gym
import sys
import random

from collections import deque
from verbal_gym.envs.gridworld.room import Room
from verbal_gym.envs.gridworld.scene import Scene


class RandomizedGridworld(gym.Env):

    # Feedback level
    Bandit, Gold, Oracle = range(3)

    def __init__(self, num_rooms=20, horizon=20, fixed=True, feedback_level="gold", goal_dist=4):
        super(RandomizedGridworld, self).__init__()

        # Action space consists of 4 actions: North, South, East and West
        self.num_actions = 4
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.observation_space = gym.spaces.Text(sys.maxsize)

        self.num_rooms = num_rooms
        self.horizon = horizon
        assert self.horizon >= 5, "Horizon must be at least 5 to allow agent to somewhat explore the world"
        self.goal_dist = goal_dist

        self.fixed = fixed

        self.docstring = "You are in a house with multiple rooms. Each room can have objects that will be visible to " \
                         "you if you are in that room. Each room can have a door along the North, South, East and " \
                         "West direction. You can follow a direction to go from one room to another. " \
                         "If there is no door along that direction, then you will remain in the room. You will start " \
                         "in a room. Your goal is to navigate to another room which has the treasure."

        if feedback_level == "bandit":
            self.feedback_level = RandomizedGridworld.Bandit

        elif feedback_level == "gold":
            self.feedback_level = RandomizedGridworld.Gold

        elif feedback_level == "oracle":
            self.feedback_level = RandomizedGridworld.Oracle

        else:
            raise AssertionError(f"Unhandled feedback level {feedback_level}")

        # Counters that may have to be reset
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
                 if self.goal_dist < len(path) < self.horizon - 5 and ngbr_room != goal_room]

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
        self.current_scene.print()

        self.current_room = self.current_scene.get_start_room()
        self.goal_prev_visited = False

        obs = self.make_room_obs(self.current_room)

        return obs

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

        if self.feedback_level == RandomizedGridworld.Bandit:
            if reward == 1.0:
                feedback = f"You succeeded! Congratulations."
            elif self.goal_prev_visited:
                feedback = f"You have already reached the treasure. Good job."
            else:
                feedback = f"You didn't succeed. Trying visiting more rooms."

        elif self.feedback_level == RandomizedGridworld.Gold:

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

        elif self.feedback_level == RandomizedGridworld.Oracle:

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
            raise AssertionError(f"Unhandled feedback level {self.feedback_level}")

        info = {
            "feedback": feedback
        }

        return next_obs, reward, done, info


if __name__ == "__main__":
    env = RandomizedGridworld(num_rooms=10, feedback_level="oracle")
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