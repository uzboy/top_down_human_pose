from lr_schedule.lr_schedule_base import LrScheduleBase


class LrScheduleStep(LrScheduleBase):

    def __init__(self, optimizer, epoch_size, cfg):
        super(LrScheduleStep, self).__init__(optimizer, epoch_size, cfg.warmup, cfg.warmup_iters, cfg.warmup_ratio)
        self.step = cfg.step
        self.gamma = cfg.gamma
        self.min_lr = cfg.min_lr

    def get_lr(self, epoch_index):
        progress = epoch_index
        if isinstance(self.step, int):
            exp = progress // self.step
        else:
            exp = len(self.step)
            for i, s in enumerate(self.step):
                if progress < s:
                    exp = i
                    break

        lr = [_lr * (self.gamma ** exp) for _lr in self.base_lr]
        if self.min_lr is not None:
            lr = max(lr, self.min_lr)
        return lr
