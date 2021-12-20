from lr_schedule.lr_schedule_step import LrScheduleStep


LRSCHEDULES = {
    "LrScheduleStep": LrScheduleStep
}


def build_lr_schedule(cfg, optimizer, epoch_size):
    return LRSCHEDULES[cfg.name](optimizer, epoch_size, cfg)

