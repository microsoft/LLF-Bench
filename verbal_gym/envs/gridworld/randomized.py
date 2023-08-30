import gym
import random

from collections import deque


class Room:

    ROOM_TYPES = ["kitchen", "bedroom", "lobby", "toilet", "balcony", "corridor", "drawing room"]
    NORTH = "north"
    EAST = "east"
    WEST = "west"
    SOUTH = "south"

    def __init__(self, room_type, room_id):

        self.room_type = room_type
        self.room_id = room_id
        self.name = f"{self.room_type}-{room_id}"

        self.doors = dict()

    def get_name(self):
        return self.name

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

    def add_door(self, other_room, dir_to):
        self.doors[dir_to] = other_room
        other_room.doors[self.opposite_direction(dir_to)] = self


class Scene:

    def __init__(self):
        self.rooms = []
        self.key_room = None

        self.room_ctr = dict()

    def create_random_empty_room(self):

        room_type = random.choice(Room.ROOM_TYPES)

        if room_type not in self.room_ctr:
            self.room_ctr[room_type] = 0

        self.room_ctr[room_type] += 1

        room = Room(room_type=room_type,
                    room_id=self.room_ctr[room_type])

        return room


class RandomizedGridworld(gym.Env):

    def __init__(self):
        super(RandomizedGridworld, self).__init__()

        # Action space consists of 4 actions: North, South, East and West
        self.num_actions = 4
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.room_types = ["kitchen", "bedroom", "lobby", "toilet", "balcony",
                           "corridor", "drawing room"]
        self.num_rooms = 10

    def make_scene(self):

        # Process of creating a scene works as follows
        # We start by creating a room
        # We add between 1-4 edges, we create new rooms and add them to the queue

        scene = Scene()
        queue = deque()

        room = scene.create_random_empty_room()

        queue.append(room)
        scene.rooms.append(room)

        while len(queue) > 0 or len(scene.rooms) < self.num_rooms:

            room = queue.popleft()

            # Sample a subset of edges from the list of available directions
            available_directions = [direction
                                    for direction in [Room.NORTH, Room.WEST, Room.SOUTH, Room.EAST]
                                    if direction not in room.doors]

            num_dir = random.randint(0, len(available_directions) - 1)
            chosen_directions = random.sample(available_directions, k=num_dir)

            for i, dir_to in enumerate(chosen_directions):

                ngbr_room = scene.create_random_empty_room()
                room.add_door(ngbr_room, dir_to)

                queue.append(ngbr_room)
                scene.rooms.append(ngbr_room)

    def reset(self):
        pass