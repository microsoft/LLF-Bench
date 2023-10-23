from collections import defaultdict
from typing import Any

class StreamLogger:
    """ Api for logging a stream of data."""

    def init(self):
        raise NotImplementedError

    def log(self, **kwargs):
        raise NotImplementedError

    @property
    def content(self) -> Any:
        raise NotImplementedError


class ListLogger(StreamLogger):
    """ Log a stream of data into a dict of lists."""
    def __init__(self):
        self._data = defaultdict(list)

    def log(self, **kwargs):
        for k,v in kwargs.items():
            self._data[k].append(v)

    @property
    def content(self):
        return self._data
