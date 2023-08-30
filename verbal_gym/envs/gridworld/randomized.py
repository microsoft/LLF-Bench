import pdb
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
        self.rooms.append(room)

        return room

    def print(self):

        for room in self.rooms:
            print(f"Room {room.get_name()}:")
            for dir_to, new_room in room.doors.items():
                print(f"\t - Taking {dir_to} path leads to {new_room.get_name()}.")
            print("\n\n")


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

        while len(queue) > 0 and len(scene.rooms) < self.num_rooms:

            room = queue.popleft()

            # Sample a subset of edges from the list of available directions
            available_directions = [direction
                                    for direction in [Room.NORTH, Room.WEST, Room.SOUTH, Room.EAST]
                                    if direction not in room.doors]

            if len(available_directions) == 0:
                # All directions from this room has been connected
                continue

            num_dir = random.randint(1, len(available_directions) - 1)
            chosen_directions = random.sample(available_directions, k=num_dir)

            for i, dir_to in enumerate(chosen_directions):

                # TODO Connect the new room not just to where it spawned from but also other rooms to create a graph
                ngbr_room = scene.create_random_empty_room()
                room.add_door(ngbr_room, dir_to)
                queue.append(ngbr_room)

        # TODO add objects in each room

        return scene

    def reset(self):

        scene = self.make_scene()
        scene.print()

        # TODO select room with key and start state

        # TODO use DFS to generate feedback

        pdb.set_trace()


if __name__ == "__main__":
    env = RandomizedGridworld()
    env.reset()
