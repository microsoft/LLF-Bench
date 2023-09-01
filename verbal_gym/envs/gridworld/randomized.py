import pdb
import gym
import random

from collections import deque


class Room:

    ROOM_TYPES = ["kitchen", "bedroom", "lobby", "toilet", "balcony", "corridor", "drawing room"]
    OBJECTS = ["lamp", "table", "couch", "television", "fridge"]
    NORTH = "north"     # Action 0
    EAST = "east"       # Action 1
    WEST = "west"       # Action 2
    SOUTH = "south"     # Action 3
    DIRECTIONS = [NORTH, EAST, WEST, SOUTH]

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

        self.doors = dict()

    def get_name(self):
        return self.name

    def get_pos(self):
        return self.pos

    def add_object(self, obj):

        if len(self.objects) < self.max_objects:
            self.objects.append(obj)
        else:
            raise AssertionError(f"Cannot add more than {self.max_objects} to {self.name}")

    @staticmethod
    def opposite_direction(dir_to):

        if dir_to == Room.NORTH:
            return Room.SOUTH

        elif dir_to == Room.SOUTH:
            return Room.NORTH

        elif dir_to == Room.WEST:
            return Room.EAST

        elif dir_to == Room.EAST:
            return Room.WEST

        else:
            raise AssertionError(f"Direction {dir_to} is unhandled.")

    def check_pos_consistentcy(self, room, other_room, dir_to):
        # An additional check to see

        room_x, room_y = self.pos
        other_room_x, other_room_y = other_room.get_pos()

        if dir_to == Room.NORTH:
            is_consistent = room_y < other_room_y
        elif dir_to == Room.SOUTH:
            is_consistent = room_y > other_room_y
        elif dir_to == Room.WEST:
            is_consistent = room_x > other_room_x
        elif dir_to == Room.EAST:
            is_consistent = room_x < other_room_x
        else:
            raise AssertionError(f"Direction {dir_to} is unhandled.")

        return is_consistent

    def get_relative_pos(self, dir_to, length=1):

        if dir_to == Room.NORTH:
            new_pos = (self.pos.x, self.pos.y + length)
        elif dir_to == Room.SOUTH:
            new_pos = (self.pos.x, self.pos.y - length)
        elif dir_to == Room.WEST:
            new_pos = (self.pos.x - length, self.pos.y)
        elif dir_to == Room.EAST:
            new_pos = (self.pos.x + length, self.pos.y)
        else:
            raise AssertionError(f"Direction {dir_to} is unhandled.")

        return new_pos

    def add_door(self, other_room, dir_to):
        self.doors[dir_to] = other_room
        other_room.doors[self.opposite_direction(dir_to)] = self


class Scene:

    def __init__(self):
        self.rooms = []
        self.key_room = None

        self.room_ctr = dict()

    def create_random_empty_room(self, pos):

        room_type = random.choice(Room.ROOM_TYPES)

        if room_type not in self.room_ctr:
            self.room_ctr[room_type] = 0

        self.room_ctr[room_type] += 1

        room = Room(room_type=room_type,
                    room_id=self.room_ctr[room_type],
                    pos=pos)
        self.rooms.append(room)

        return room

    def get_room(self, i):
        return self.rooms[i]

    def num_rooms(self):
        return len(self.rooms)

    def print(self):

        for room in self.rooms:
            print(f"Room {room.get_name()}:")
            for dir_to, new_room in room.doors.items():
                print(f"\t - Taking {dir_to} path leads to {new_room.get_name()}.")
            print("\n\n")


class RandomizedGridworld(gym.Env):

    def __init__(self, fixed):
        super(RandomizedGridworld, self).__init__()

        # Action space consists of 4 actions: North, South, East and West
        self.num_actions = 4
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.num_rooms = 10

        self.fixed = fixed

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
            available_directions = [direction for direction in Room.DIRECTIONS if direction not in room.doors]

            if len(available_directions) == 0:
                # All directions from this room has been connected
                continue

            num_dir = random.randint(1, len(available_directions) - 1)
            chosen_directions = random.sample(available_directions, k=num_dir)

            for i, dir_to in enumerate(chosen_directions):

                # TODO Connect the new room not just to where it spawned from but also other rooms to create a graph
                new_pos = room.get_relative_pos(dir_to, length=1)
                ngbr_room = scene.create_random_empty_room(pos=new_pos)
                room.add_door(ngbr_room, dir_to)
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
        # Add key in a room at least k steps away

        return scene

    def reset(self):

        scene = self.make_scene()
        scene.print()

        # TODO select room with key and start state

        # TODO use DFS to generate feedback

        pdb.set_trace()

    def step(self, action):
        pass


if __name__ == "__main__":
    env = RandomizedGridworld()
    env.reset()
