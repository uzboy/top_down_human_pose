from abc import abstractmethod


class LrScheduleBase:

    def __init__(self, optimizer, epoch_size, warmup=None, warmup_iters=0, warmup_ratio=0.1):
        self.optimizer = optimizer
        self.epoch_size = epoch_size
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio

        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
        self.base_lr = [group['initial_lr'] for group in optimizer.param_groups]

    def _set_lr(self, lr_groups):
        for param_group, lr in zip(self.optimizer.param_groups, lr_groups):
            param_group['lr'] = lr

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.base_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.base_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio ** (1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.base_lr]
        return warmup_lr

    def _do_update_without_warmup(self, epoch_index, iter_index):
        if iter_index == 1:
            base_lr = self.get_lr(epoch_index)
            self._set_lr(base_lr)
    
    def _do_update_with_warmup(self, epoch_index, iter_index):
        cur_iter = epoch_index * self.epoch_size + iter_index
        if cur_iter <= self.warmup_iters:
            warmup_lr = self.get_warmup_lr(cur_iter)
            self._set_lr(warmup_lr)
        elif iter_index == 1:
            base_lr = self.get_lr(epoch_index)
            self._set_lr(base_lr)
    
    def update_lr(self, epoch_index, iter_index):
        if self.warmup is None:
            self._do_update_without_warmup(epoch_index, iter_index)
        else:
            self._do_update_with_warmup(epoch_index, iter_index)

    @abstractmethod
    def get_lr(self, epoch_index):
        raise NotImplementedError
