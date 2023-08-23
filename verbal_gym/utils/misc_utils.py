import re


# print with colors (modified from Huihan's lflf)
def print_color(message, color=None):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    print(f"{colors.get(color, '')}{message}\033[0m")  # Default to no color if invalid color is provided


def extract_int(txt):
    return [int(s) for s in txt.split() if s.isdigit()]
