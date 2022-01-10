import argparse
from utils.utils import get_logger
from torch.utils.data import DataLoader
from config.build_config import build_config
from dataset.build_dataset import build_dataset
from models.builder_model import build_model
from builder.build_optimizer import build_optimizer
from trainer.simple_trainer import SimpleTrainer as Trainer
from lr_schedule.build_lr_schedule import build_lr_schedule
from data_process.data_collection import build_collect_func


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('--config', default="MobileNetV2Blaze", type=str, help='train config file path')
    args = parser.parse_args()
    cfg = build_config(args.config)

    logger = get_logger(cfg.logfile)
    model = build_model(cfg.model)

    train_dataset = build_dataset(cfg.data)
    collate_fn = cfg.data.get("collate_fn", None)
    if collate_fn is not None:
        collate_fn = build_collect_func(collate_fn)

    train_data_loader = DataLoader(train_dataset,
                                                                        shuffle=True,
                                                                        batch_size=cfg.data.batch_size,
                                                                        pin_memory=cfg.data.get("pin_memory", False),
                                                                        num_workers=cfg.data.get("num_workers", 1),
                                                                        drop_last=True,
                                                                        collate_fn=collate_fn)

    optimizer = build_optimizer(cfg.optimizer, model)
    lr_schedule = build_lr_schedule(cfg.lr_schedule, optimizer, len(train_dataset))
    epoch_trainer = Trainer(model,
                                                      train_data_loader,
                                                      optimizer,
                                                      cfg.model.get("device", "cpu"),
                                                      lr_schedule,
                                                      logger, cfg.get("logger_freq", 10),
                                                      cfg.model.head.get("update_loss", "total_loss"))

    start_epoch = cfg.get("start_epoch", 0)
    end_epoch = cfg.end_epoch
    save_ckps_freq = cfg.get("save_ckps_freq", 1)
    for ep in range(start_epoch, end_epoch):
        if train_dataset is not None:
            is_save_ckps = ((ep % save_ckps_freq == 0) or (ep == end_epoch - 1))
            epoch_trainer(ep, is_save_ckps)
