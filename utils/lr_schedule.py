# from abc import abstractmethod
# import numbers
# from math import cos, pi




# class FixedLrUpdaterHook(LrUpdaterHook):

#     def __init__(self, optimizer, epoch_size, warmup=None, warmup_iters=0, warmup_ratio=0.1):
#         super(FixedLrUpdaterHook, self).__init__(optimizer, epoch_size, warmup, warmup_iters, warmup_ratio)

#     def get_lr(self, epoch_index):
#         return self.base_lr


# class ExpLrUpdaterHook(LrUpdaterHook):

#     def __init__(self, gamma, optimizer, epoch_size, warmup=None, warmup_iters=0, warmup_ratio=0.1):
#         self.gamma = gamma
#         super(ExpLrUpdaterHook, self).__init__(optimizer, epoch_size, warmup, warmup_iters, warmup_ratio)

#     def get_lr(self, epoch_index):
#         return self.base_lr * self.gamma** epoch_index


# class PolyLrUpdaterHook(LrUpdaterHook):

#     def __init__(self, optimizer, max_epochs, epoch_size, warmup=None, warmup_iters=0, warmup_ratio=0.1, power=1., min_lr=0.):
#         self.power = power
#         self.min_lr = min_lr
#         self.max_epochs = max_epochs
#         super(PolyLrUpdaterHook, self).__init__(optimizer, epoch_size, warmup, warmup_iters, warmup_ratio)

#     def get_lr(self, epoch_index):
#         coeff = (1 - epoch_index / self.max_epochs) ** self.power
#         return (self.base_lr - self.min_lr) * coeff + self.min_lr


# class InvLrUpdaterHook(LrUpdaterHook):

#     def __init__(self, optimizer, gamma, epoch_size, warmup=None, warmup_iters=0, warmup_ratio=0.1, power=1.):
#         self.gamma = gamma
#         self.power = power
#         super(InvLrUpdaterHook, self).__init__(optimizer, epoch_size, warmup, warmup_iters, warmup_ratio)

#     def get_lr(self, epoch_index):
#         return self.base_lr * (1 + self.gamma * epoch_index)**(-self.power)


# class CosineAnnealingLrUpdaterHook(LrUpdaterHook):

#     def __init__(self, min_lr=None, min_lr_ratio=None, **kwargs):
#         assert (min_lr is None) ^ (min_lr_ratio is None)
#         self.min_lr = min_lr
#         self.min_lr_ratio = min_lr_ratio
#         super(CosineAnnealingLrUpdaterHook, self).__init__(**kwargs)

#     def get_lr(self, runner, base_lr):
#         if self.by_epoch:
#             progress = runner.epoch
#             max_progress = runner.max_epochs
#         else:
#             progress = runner.iter
#             max_progress = runner.max_iters

#         if self.min_lr_ratio is not None:
#             target_lr = base_lr * self.min_lr_ratio
#         else:
#             target_lr = self.min_lr
#         return annealing_cos(base_lr, target_lr, progress / max_progress)


# class FlatCosineAnnealingLrUpdaterHook(LrUpdaterHook):
#     """Flat + Cosine lr schedule.

#     Modified from https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L128 # noqa: E501

#     Args:
#         start_percent (float): When to start annealing the learning rate
#             after the percentage of the total training steps.
#             The value should be in range [0, 1).
#             Default: 0.75
#         min_lr (float, optional): The minimum lr. Default: None.
#         min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
#             Either `min_lr` or `min_lr_ratio` should be specified.
#             Default: None.
#     """

#     def __init__(self,
#                  start_percent=0.75,
#                  min_lr=None,
#                  min_lr_ratio=None,
#                  **kwargs):
#         assert (min_lr is None) ^ (min_lr_ratio is None)
#         if start_percent < 0 or start_percent > 1 or not isinstance(
#                 start_percent, float):
#             raise ValueError(
#                 'expected float between 0 and 1 start_percent, but '
#                 f'got {start_percent}')
#         self.start_percent = start_percent
#         self.min_lr = min_lr
#         self.min_lr_ratio = min_lr_ratio
#         super(FlatCosineAnnealingLrUpdaterHook, self).__init__(**kwargs)

#     def get_lr(self, runner, base_lr):
#         if self.by_epoch:
#             start = round(runner.max_epochs * self.start_percent)
#             progress = runner.epoch - start
#             max_progress = runner.max_epochs - start
#         else:
#             start = round(runner.max_iters * self.start_percent)
#             progress = runner.iter - start
#             max_progress = runner.max_iters - start

#         if self.min_lr_ratio is not None:
#             target_lr = base_lr * self.min_lr_ratio
#         else:
#             target_lr = self.min_lr

#         if progress < 0:
#             return base_lr
#         else:
#             return annealing_cos(base_lr, target_lr, progress / max_progress)


# class CosineRestartLrUpdaterHook(LrUpdaterHook):
#     """Cosine annealing with restarts learning rate scheme.

#     Args:
#         periods (list[int]): Periods for each cosine anneling cycle.
#         restart_weights (list[float], optional): Restart weights at each
#             restart iteration. Default: [1].
#         min_lr (float, optional): The minimum lr. Default: None.
#         min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
#             Either `min_lr` or `min_lr_ratio` should be specified.
#             Default: None.
#     """

#     def __init__(self,
#                  periods,
#                  restart_weights=[1],
#                  min_lr=None,
#                  min_lr_ratio=None,
#                  **kwargs):
#         assert (min_lr is None) ^ (min_lr_ratio is None)
#         self.periods = periods
#         self.min_lr = min_lr
#         self.min_lr_ratio = min_lr_ratio
#         self.restart_weights = restart_weights
#         assert (len(self.periods) == len(self.restart_weights)
#                 ), 'periods and restart_weights should have the same length.'
#         super(CosineRestartLrUpdaterHook, self).__init__(**kwargs)

#         self.cumulative_periods = [
#             sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
#         ]

#     def get_lr(self, runner, base_lr):
#         if self.by_epoch:
#             progress = runner.epoch
#         else:
#             progress = runner.iter

#         if self.min_lr_ratio is not None:
#             target_lr = base_lr * self.min_lr_ratio
#         else:
#             target_lr = self.min_lr

#         idx = get_position_from_periods(progress, self.cumulative_periods)
#         current_weight = self.restart_weights[idx]
#         nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
#         current_periods = self.periods[idx]

#         alpha = min((progress - nearest_restart) / current_periods, 1)
#         return annealing_cos(base_lr, target_lr, alpha, current_weight)


# def get_position_from_periods(iteration, cumulative_periods):
#     """Get the position from a period list.

#     It will return the index of the right-closest number in the period list.
#     For example, the cumulative_periods = [100, 200, 300, 400],
#     if iteration == 50, return 0;
#     if iteration == 210, return 2;
#     if iteration == 300, return 3.

#     Args:
#         iteration (int): Current iteration.
#         cumulative_periods (list[int]): Cumulative period list.

#     Returns:
#         int: The position of the right-closest number in the period list.
#     """
#     for i, period in enumerate(cumulative_periods):
#         if iteration < period:
#             return i
#     raise ValueError(f'Current iteration {iteration} exceeds '
#                      f'cumulative_periods {cumulative_periods}')


# class CyclicLrUpdaterHook(LrUpdaterHook):
#     """Cyclic LR Scheduler.

#     Implement the cyclical learning rate policy (CLR) described in
#     https://arxiv.org/pdf/1506.01186.pdf

#     Different from the original paper, we use cosine annealing rather than
#     triangular policy inside a cycle. This improves the performance in the
#     3D detection area.

#     Args:
#         by_epoch (bool): Whether to update LR by epoch.
#         target_ratio (tuple[float]): Relative ratio of the highest LR and the
#             lowest LR to the initial LR.
#         cyclic_times (int): Number of cycles during training
#         step_ratio_up (float): The ratio of the increasing process of LR in
#             the total cycle.
#         anneal_strategy (str): {'cos', 'linear'}
#             Specifies the annealing strategy: 'cos' for cosine annealing,
#             'linear' for linear annealing. Default: 'cos'.
#     """

#     def __init__(self,
#                  by_epoch=False,
#                  target_ratio=(10, 1e-4),
#                  cyclic_times=1,
#                  step_ratio_up=0.4,
#                  anneal_strategy='cos',
#                  **kwargs):
#         if isinstance(target_ratio, float):
#             target_ratio = (target_ratio, target_ratio / 1e5)
#         elif isinstance(target_ratio, tuple):
#             target_ratio = (target_ratio[0], target_ratio[0] / 1e5) \
#                 if len(target_ratio) == 1 else target_ratio
#         else:
#             raise ValueError('target_ratio should be either float '
#                              f'or tuple, got {type(target_ratio)}')

#         assert len(target_ratio) == 2, \
#             '"target_ratio" must be list or tuple of two floats'
#         assert 0 <= step_ratio_up < 1.0, \
#             '"step_ratio_up" must be in range [0,1)'

#         self.target_ratio = target_ratio
#         self.cyclic_times = cyclic_times
#         self.step_ratio_up = step_ratio_up
#         self.lr_phases = []  # init lr_phases
#         # validate anneal_strategy
#         if anneal_strategy not in ['cos', 'linear']:
#             raise ValueError('anneal_strategy must be one of "cos" or '
#                              f'"linear", instead got {anneal_strategy}')
#         elif anneal_strategy == 'cos':
#             self.anneal_func = annealing_cos
#         elif anneal_strategy == 'linear':
#             self.anneal_func = annealing_linear

#         assert not by_epoch, \
#             'currently only support "by_epoch" = False'
#         super(CyclicLrUpdaterHook, self).__init__(by_epoch, **kwargs)

#     def before_run(self, runner):
#         super(CyclicLrUpdaterHook, self).before_run(runner)
#         # initiate lr_phases
#         # total lr_phases are separated as up and down
#         max_iter_per_phase = runner.max_iters // self.cyclic_times
#         iter_up_phase = int(self.step_ratio_up * max_iter_per_phase)
#         self.lr_phases.append(
#             [0, iter_up_phase, max_iter_per_phase, 1, self.target_ratio[0]])
#         self.lr_phases.append([
#             iter_up_phase, max_iter_per_phase, max_iter_per_phase,
#             self.target_ratio[0], self.target_ratio[1]
#         ])

#     def get_lr(self, runner, base_lr):
#         curr_iter = runner.iter
#         for (start_iter, end_iter, max_iter_per_phase, start_ratio,
#              end_ratio) in self.lr_phases:
#             curr_iter %= max_iter_per_phase
#             if start_iter <= curr_iter < end_iter:
#                 progress = curr_iter - start_iter
#                 return self.anneal_func(base_lr * start_ratio,
#                                         base_lr * end_ratio,
#                                         progress / (end_iter - start_iter))


# class OneCycleLrUpdaterHook(LrUpdaterHook):
#     """One Cycle LR Scheduler.

#     The 1cycle learning rate policy changes the learning rate after every
#     batch. The one cycle learning rate policy is described in
#     https://arxiv.org/pdf/1708.07120.pdf

#     Args:
#         max_lr (float or list): Upper learning rate boundaries in the cycle
#             for each parameter group.
#         total_steps (int, optional): The total number of steps in the cycle.
#             Note that if a value is not provided here, it will be the max_iter
#             of runner. Default: None.
#         pct_start (float): The percentage of the cycle (in number of steps)
#             spent increasing the learning rate.
#             Default: 0.3
#         anneal_strategy (str): {'cos', 'linear'}
#             Specifies the annealing strategy: 'cos' for cosine annealing,
#             'linear' for linear annealing.
#             Default: 'cos'
#         div_factor (float): Determines the initial learning rate via
#             initial_lr = max_lr/div_factor
#             Default: 25
#         final_div_factor (float): Determines the minimum learning rate via
#             min_lr = initial_lr/final_div_factor
#             Default: 1e4
#         three_phase (bool): If three_phase is True, use a third phase of the
#             schedule to annihilate the learning rate according to
#             final_div_factor instead of modifying the second phase (the first
#             two phases will be symmetrical about the step indicated by
#             pct_start).
#             Default: False
#     """

#     def __init__(self,
#                  max_lr,
#                  total_steps=None,
#                  pct_start=0.3,
#                  anneal_strategy='cos',
#                  div_factor=25,
#                  final_div_factor=1e4,
#                  three_phase=False,
#                  **kwargs):
#         # validate by_epoch, currently only support by_epoch = False
#         if 'by_epoch' not in kwargs:
#             kwargs['by_epoch'] = False
#         else:
#             assert not kwargs['by_epoch'], \
#                 'currently only support "by_epoch" = False'
#         if not isinstance(max_lr, (numbers.Number, list, dict)):
#             raise ValueError('the type of max_lr must be the one of list or '
#                              f'dict, but got {type(max_lr)}')
#         self._max_lr = max_lr
#         if total_steps is not None:
#             if not isinstance(total_steps, int):
#                 raise ValueError('the type of total_steps must be int, but'
#                                  f'got {type(total_steps)}')
#             self.total_steps = total_steps
#         # validate pct_start
#         if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
#             raise ValueError('expected float between 0 and 1 pct_start, but '
#                              f'got {pct_start}')
#         self.pct_start = pct_start
#         # validate anneal_strategy
#         if anneal_strategy not in ['cos', 'linear']:
#             raise ValueError('anneal_strategy must be one of "cos" or '
#                              f'"linear", instead got {anneal_strategy}')
#         elif anneal_strategy == 'cos':
#             self.anneal_func = annealing_cos
#         elif anneal_strategy == 'linear':
#             self.anneal_func = annealing_linear
#         self.div_factor = div_factor
#         self.final_div_factor = final_div_factor
#         self.three_phase = three_phase
#         self.lr_phases = []  # init lr_phases
#         super(OneCycleLrUpdaterHook, self).__init__(**kwargs)

#     def before_run(self, runner):
#         if hasattr(self, 'total_steps'):
#             total_steps = self.total_steps
#         else:
#             total_steps = runner.max_iters
#         if total_steps < runner.max_iters:
#             raise ValueError(
#                 'The total steps must be greater than or equal to max '
#                 f'iterations {runner.max_iters} of runner, but total steps '
#                 f'is {total_steps}.')

#         if isinstance(runner.optimizer, dict):
#             self.base_lr = {}
#             for k, optim in runner.optimizer.items():
#                 _max_lr = format_param(k, optim, self._max_lr)
#                 self.base_lr[k] = [lr / self.div_factor for lr in _max_lr]
#                 for group, lr in zip(optim.param_groups, self.base_lr[k]):
#                     group.setdefault('initial_lr', lr)
#         else:
#             k = type(runner.optimizer).__name__
#             _max_lr = format_param(k, runner.optimizer, self._max_lr)
#             self.base_lr = [lr / self.div_factor for lr in _max_lr]
#             for group, lr in zip(runner.optimizer.param_groups, self.base_lr):
#                 group.setdefault('initial_lr', lr)

#         if self.three_phase:
#             self.lr_phases.append(
#                 [float(self.pct_start * total_steps) - 1, 1, self.div_factor])
#             self.lr_phases.append([
#                 float(2 * self.pct_start * total_steps) - 2, self.div_factor, 1
#             ])
#             self.lr_phases.append(
#                 [total_steps - 1, 1, 1 / self.final_div_factor])
#         else:
#             self.lr_phases.append(
#                 [float(self.pct_start * total_steps) - 1, 1, self.div_factor])
#             self.lr_phases.append(
#                 [total_steps - 1, self.div_factor, 1 / self.final_div_factor])

#     def get_lr(self, runner, base_lr):
#         curr_iter = runner.iter
#         start_iter = 0
#         for i, (end_iter, start_lr, end_lr) in enumerate(self.lr_phases):
#             if curr_iter <= end_iter:
#                 pct = (curr_iter - start_iter) / (end_iter - start_iter)
#                 lr = self.anneal_func(base_lr * start_lr, base_lr * end_lr,
#                                       pct)
#                 break
#             start_iter = end_iter
#         return lr


# def annealing_cos(start, end, factor, weight=1):
#     """Calculate annealing cos learning rate.

#     Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
#     percentage goes from 0.0 to 1.0.

#     Args:
#         start (float): The starting learning rate of the cosine annealing.
#         end (float): The ending learing rate of the cosine annealing.
#         factor (float): The coefficient of `pi` when calculating the current
#             percentage. Range from 0.0 to 1.0.
#         weight (float, optional): The combination factor of `start` and `end`
#             when calculating the actual starting learning rate. Default to 1.
#     """
#     cos_out = cos(pi * factor) + 1
#     return end + 0.5 * weight * (start - end) * cos_out


# def annealing_linear(start, end, factor):
#     """Calculate annealing linear learning rate.

#     Linear anneal from `start` to `end` as percentage goes from 0.0 to 1.0.

#     Args:
#         start (float): The starting learning rate of the linear annealing.
#         end (float): The ending learing rate of the linear annealing.
#         factor (float): The coefficient of `pi` when calculating the current
#             percentage. Range from 0.0 to 1.0.
#     """
#     return start + (end - start) * factor


# def format_param(name, optim, param):
#     if isinstance(param, numbers.Number):
#         return [param] * len(optim.param_groups)
#     elif isinstance(param, (list, tuple)):  # multi param groups
#         if len(param) != len(optim.param_groups):
#             raise ValueError(f'expected {len(optim.param_groups)} '
#                              f'values for {name}, got {len(param)}')
#         return param
#     else:  # multi optimizers
#         if name not in param:
#             raise KeyError(f'{name} is not found in {param.keys()}')
#         return param[name]
