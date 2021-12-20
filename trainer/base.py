from abc import abstractmethod


class TrainBase:

    def __init__(self, model, data_loader, optimizer, device, lr_schedule, logger, logger_freq):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.lr_schedule = lr_schedule
        self.logger = logger
        self.logger_freq = logger_freq

    @abstractmethod
    def train_one_epoch(self, epoch_index):
        ""

    def __call__(self, epoch_index):
        self.train_one_epoch(epoch_index)
