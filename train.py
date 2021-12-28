import os
import argparse
from utils.utils import get_logger
from utils.utils import save_checkpoint
from torch.utils.data import DataLoader
from config.build_config import build_config
from dataset.build_dataset import build_dataset
from models.builder_model import build_model
from builder.build_optimizer import build_optimizer
from trainer.simple_trainer import SimpleTrainer as Trainer
from lr_schedule.build_lr_schedule import build_lr_schedule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config', default="MobileNetV2", type=str, help='train config file path')
    args = parser.parse_args()
    cfg = build_config(args.config)

    logger = get_logger(cfg.logfile)
    model = build_model(cfg.model)

    train_dataset = build_dataset(cfg.data.train)
    train_data_loader = DataLoader(train_dataset,
                                                                        shuffle=True,
                                                                        batch_size=cfg.data.train.batch_size,
                                                                        pin_memory=cfg.data.train.pin_memory,
                                                                        num_workers=cfg.data.train.num_workers,
                                                                        drop_last=True)

    optimizer = build_optimizer(cfg.optimizer, model)
    lr_schedule = build_lr_schedule(cfg.lr_schedule, optimizer, len(train_dataset))

    if not hasattr(cfg, "loss_num"):
        loss_num = 1
    else:
        loss_num = cfg.loss_num
    epoch_trainer = Trainer(model, train_data_loader, optimizer, cfg.device, lr_schedule, logger, cfg.logger_freq, loss_num)

    for ep in range(cfg.start_epoch, cfg.end_epoch):
        if train_dataset is not None:
            epoch_trainer(ep)
            if ep % cfg.save_ckps_freq == 0:
                save_checkpoint(model, os.path.join(cfg.mode_root, cfg.model_pth_name))
