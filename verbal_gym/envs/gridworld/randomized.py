import pdb
import gym
import random

from collections import deque


class Room:

    ROOM_TYPES = ["kitchen", "bedroom", "lobby", "toilet", "balcony", "corridor", "drawing room"]
    OBJECTS = ["lamp", "table", "couch", "television", "fridge"]
    goal = "treasure"

    def __init__(self, room_type, room_id, pos, max_objects=2):
        """
        :param room_type: type of the room from Room.ROOM_TYPES
        :param room_id: a number to distinguish between multiple rooms with the same ID
        :param pos: (x, y) pair where x is the horizontal axis with west towards -infinity and east towards +infinity
                    and y is the vertical axis with north towards +infinity and south towards -infinity.
        """

        assert room_type in Room.ROOM_TYPES, \
            f"room_type {room_type} must be one of the following types {Room.ROOM_TYPES}"
        assert type(pos) == tuple and len(pos) == 2, "Position is a tuple containing (x, y)"

        self.room_type = room_type
        self.room_id = room_id
        self.pos = pos
        self.name = f"{self.room_type}-{room_id}"

        self.max_objects = max_objects
        self.objects = []

    def get_name(self):
        return self.name

    def get_objects(self):
        return self.objects

    def describe_room(self):
        s = f"You are in {self.name} room. "

        if len(self.objects) > 0:
            s += "This room has following objects: " + ",".join(self.objects) + ". "

        return s

    def get_pos(self):
        return self.pos

    def add_goal(self):
        self.objects.append(Room.goal)

    def add_object(self, obj):

        if len(self.objects) < self.max_objects:
            self.objects.append(obj)
        else:
            raise AssertionError(f"Cannot add more than {self.max_objects} to {self.name}")


class Scene:

    NORTH = "north"  # Action 0
    EAST = "east"    # Action 1
    WEST = "west"    # Action 2
    SOUTH = "south"  # Action 3
    DIRECTIONS = [NORTH, EAST, WEST, SOUTH]

    def __init__(self):
        self.rooms = []

        self.start_room = None
        self.key_room = None

        self.room_ctr = dict()
        self.doors = dict()

        # Run BFS
        self.bfs_path = dict()

    def get_add_start_room(self, start_room):
        self.start_room = start_room

    def get_add_goal_room(self, goal_room):
        self.key_room = goal_room

    def get_room(self, i):
        return self.rooms[i]

    def num_rooms(self):
        return len(self.rooms)

    def get_rooms(self):
        return self.rooms

    def get_start_room(self):
        return self.start_room

    def check_room_door(self, room, dir_to):

        if room in self.doors and dir_to in self.doors[room]:
            return self.doors[room][dir_to]
        else:
            return None

    def get_room_doors_description(self, room):

        s = " ".join([f"You have a door to the {dir_to} of you that takes you to the {ngbr_room.get_name()} room."
                     for dir_to, ngbr_room in self.doors[room].items()])
        return s

    def create_random_empty_room(self, pos):

        room_type = random.choice(Room.ROOM_TYPES)

        if room_type not in self.room_ctr:
            self.room_ctr[room_type] = 0

        self.room_ctr[room_type] += 1

        room = Room(room_type=room_type,
                    room_id=self.room_ctr[room_type],
                    pos=pos)
        self.rooms.append(room)
        self.doors[room] = dict()

        return room

    @staticmethod
    def opposite_direction(dir_to):

        if dir_to == Scene.NORTH:
            return Scene.SOUTH

        elif dir_to == Scene.SOUTH:
            return Scene.NORTH

        elif dir_to == Scene.WEST:
            return Scene.EAST

        elif dir_to == Scene.EAST:
            return Scene.WEST

        else:
            raise AssertionError(f"Direction {dir_to} is unhandled.")

    @staticmethod
    def check_pos_consistentcy(room, other_room, dir_to):
        # An additional check to see

        room_x, room_y = room.pos
        other_room_x, other_room_y = other_room.get_pos()

        if dir_to == Scene.NORTH:
            is_consistent = room_y < other_room_y

        elif dir_to == Scene.SOUTH:
            is_consistent = room_y > other_room_y

        elif dir_to == Scene.WEST:
            is_consistent = room_x > other_room_x

        elif dir_to == Scene.EAST:
            is_consistent = room_x < other_room_x

        else:
            raise AssertionError(f"Direction {dir_to} is unhandled.")

        return is_consistent

    @staticmethod
    def get_relative_pos(room, dir_to, length=1):

        room_x, room_y = room.pos

        if dir_to == Scene.NORTH:
            new_pos = (room_x, room_y + length)

        elif dir_to == Scene.SOUTH:
            new_pos = (room_x, room_y - length)

        elif dir_to == Scene.WEST:
            new_pos = (room_x - length, room_y)

        elif dir_to == Scene.EAST:
            new_pos = (room_x + length, room_y)

        else:
            raise AssertionError(f"Direction {dir_to} is unhandled.")

        return new_pos

    def add_door(self, room, dir_to, other_room):
        self.doors[room][dir_to] = other_room
        self.doors[other_room][self.opposite_direction(dir_to)] = room

    def start_bfs(self, start_room):

        queue = deque([])

        queue.append(start_room)
        self.bfs_path[start_room] = []

        while len(queue) > 0:

            room = queue.popleft()

            for dir_to, ngbr_room in self.doors[room].items():

                if ngbr_room not in self.bfs_path:

                    queue.append(ngbr_room)
                    path = list(self.bfs_path[room])
                    path.append((dir_to, ngbr_room))
                    self.bfs_path[ngbr_room] = path

        return self.bfs_path

    def get_gold_action(self, room):

        # This is a path from this goal to this room
        # Path consists of [(action-1, room-1), (action-2, room-2), ...., (action-k, room-k)] where room-k=room
        path = self.bfs_path[room]

        if len(path) == 0:  # No action to take
            gold_direction = None
        else:
            direction, room_ = path[-1]
            assert room_ == room
            # We need to in opposite direction
            gold_direction = self.opposite_direction(direction)

        return gold_direction

    def print(self):

        print(f"Start room {self.start_room.get_name()} and Key room {self.key_room.get_name()}\n")

        for room in self.rooms:
            obj_names = ", ".join(room.get_objects())
            print(f"Room {room.get_name()}: containing objects {obj_names}")
            for dir_to, new_room in room.doors.items():
                print(f"\t - Taking {dir_to} path leads to {new_room.get_name()}.")
            print("\n\n")


class RandomizedGridworld(gym.Env):

    # Feedback level
    Bandit, Gold = range(2)

    def __init__(self, fixed=True, feedback_level="gold"):
        super(RandomizedGridworld, self).__init__()

        # Action space consists of 4 actions: North, South, East and West
        self.num_actions = 4
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.num_rooms = 10

        self.fixed = fixed

        self.docstring = "You are in a house with multiple rooms. Each room can have objects that will be visible to " \
                         "you if you are in that room. Each room can have a door along the North, South, East and " \
                         "West direction. You can follow a direction to go from one room to another. " \
                         "If there is no door along that direction, then you will remain in the room. You will start " \
                         "in a room. Your goal is to navigate to another room which has the treasure."

        self.horizon = 20
        assert self.horizon >= 5, "Horizon must be at least 5 to allow agent to somewhat explore the world"

        if feedback_level == "bandit":
            self.feedback_level = RandomizedGridworld.Bandit

        elif feedback_level == "gold":
            self.feedback_level = RandomizedGridworld.Gold

        else:
            raise AssertionError(f"Unhandled feedback level {feedback_level}")

        # Counters that may have to be reset
        self.current_timestep = 0.0
        self.current_scene = None
        self.current_room = None
        self.goal_not_prev_visited = False

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
            available_directions = [direction for direction in Scene.DIRECTIONS if direction not in room.doors]

            if len(available_directions) == 0:
                # All directions from this room has been connected
                continue

            num_dir = random.randint(1, len(available_directions) - 1)
            chosen_directions = random.sample(available_directions, k=num_dir)

            for i, dir_to in enumerate(chosen_directions):

                # TODO Connect the new room not just to where it spawned from but also other rooms to create a graph
                new_pos = room.get_relative_pos(dir_to, length=1)
                ngbr_room = scene.create_random_empty_room(pos=new_pos)
                scene.add_door(room=room,
                               dir_to=dir_to,
                               other_room=ngbr_room)
                queue.append(ngbr_room)

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
        start_room = random.choice(rooms)
        scene.get_add_start_room(start_room=start_room)

        # Do DFS
        scene.start_bfs(start_room)

        # Add key in a room at least k steps away
        k = 4
        rooms = [ngbr_room for ngbr_room, path in scene.bfs_path.items() if k < len(path) < self.horizon - 5
                 and ngbr_room != start_room]
        goal_room = random.choice(rooms)
        goal_room.add_goal()
        scene.get_add_goal_room(goal_room=goal_room)

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
        self.goal_not_prev_visited = False

        obs = self.make_room_obs(self.current_room)

        return obs

    def step(self, action):

        if self.current_timestep > self.horizon:
            raise AssertionError("Horizon exhausted.")

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
        reward = 1.0 if self.current_room == self.current_scene.goal_room and self.goal_not_prev_visited else 0.0
        self.goal_not_prev_visited = self.current_room == self.current_scene.goal_room

        # Update the counter and compute done
        self.current_timestep += 1
        done = self.current_timestep == self.horizon

        if self.feedback_level == RandomizedGridworld.Bandit:
            if reward == 1.0:
                feedback = f"You succeeded! Congratulations."
            else:
                feedback = f"You didn't succeed. Trying visiting more rooms."

        elif self.feedback_level == RandomizedGridworld.Gold:
            gold_action = self.current_scene.get_gold_action(self.current_room)
            assert gold_action is not None, "Gold action cannot be None since we are not in the goal room."
            feedback = f"You tried to go in the direction where there is no room. " \
                       f"Consider taking the {gold_action} direction"
        else:
            raise AssertionError(f"Unhandled feedback level {self.feedback_level}")

        info = {
            "feedback": feedback
        }

        return next_obs, reward, done, info


if __name__ == "__main__":
    env = RandomizedGridworld()
    env.reset()
