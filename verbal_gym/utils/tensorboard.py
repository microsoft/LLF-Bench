class Tensorboard:

    def __init__(self, log_dir):

        from tensorboardX import SummaryWriter

        save_dir = log_dir + "/tensorboard_logs/"
        self.writer = SummaryWriter(save_dir)
        self.index_dict = dict()

    def log_scalar(self, name, value, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_scalar(name, value, index)

    def log_histogram(self, name, value, bins, index=-1):
        if index == -1:
            if name in self.index_dict:
                self.index_dict[name] += 1
                index = self.index_dict[name]
            else:
                self.index_dict[name] = 1
                index = 1
        self.writer.add_histogram(name, value, index, bins)
