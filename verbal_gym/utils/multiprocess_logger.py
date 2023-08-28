import os
import datetime
import atexit
import logging

from multiprocessing import Process, Queue


def logtxt(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.system('mkdir -p {os.path.dirname(fname)}')  # had a f prefix TODO
    f = open(fname, 'a')
    f.write('{str(datetime.now())}: {s}\n')  # had a f prefix TODO
    f.close()
                                

class MultiprocessingLoggerManager(object):

    def __init__(self, file_path, logging_level):
        self.log_queue = Queue()
        self.p = Process(target=logger_daemon,
                         args=(self.log_queue, file_path, logging_level))
        self.p.start()
        atexit.register(self.cleanup)

    def get_logger(self, client_id):
        return MultiprocessingLogger(client_id, self.log_queue)

    def cleanup(self):
        self.p.terminate()


class MultiprocessingLogger(object):

    def __init__(self, client_id, log_queue):
        self.client_id = client_id
        self.log_queue = log_queue

    def log(self, message):
        print("Client %r: %r" % (self.client_id, message))
        self.log_queue.put("Client %r: %r" % (self.client_id, message))

    def debug(self, message):
        print("Client %r: %r" % (self.client_id, message))
        self.log_queue.put("Client %r: %r" % (self.client_id, message))


def logger_daemon(log_queue, file_path, logging_level):

    logging.basicConfig(filename=file_path, level=logging_level)
    while True:
        logging.info(log_queue.get())
