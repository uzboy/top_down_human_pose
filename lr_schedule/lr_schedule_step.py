from lr_schedule.lr_schedule_base import LrScheduleBase


class LrScheduleStep(LrScheduleBase):

    def __init__(self, optimizer, epoch_size, cfg):
        super(LrScheduleStep, self).__init__(optimizer, epoch_size,
                                                                                     cfg.get("warmup", None),
                                                                                     cfg.get("warmup_iters", 500),
                                                                                     cfg.get("warmup_ratio", 0.001))
        self.step = cfg.step
        self.gamma = cfg.get("gamma", 0.1)
        self.min_lr = cfg.get("min_lr", 0)

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

        lr = [max(_lr * (self.gamma ** exp), self.min_lr) for _lr in self.base_lr]
        return lr
