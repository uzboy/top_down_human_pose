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
    def train_one_epoch(self, epoch_index, save_ckps):
        ""

    def get_loss(self, loss_inputs):
        if hasattr(self.model, "module"):
            loss = self.model.module.get_loss(loss_inputs)
        else:
            loss = self.model.get_loss(loss_inputs)
        return loss

    def save_ckps(self, epoch_index):
        if hasattr(self.model, "module"):
            self.model.module.save_ckps(epoch_index)
        else:
            self.model.save_ckps(epoch_index)

    def __call__(self, epoch_index, save_ckps):
        self.train_one_epoch(epoch_index, save_ckps)
