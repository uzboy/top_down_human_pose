import torch


def build_optimizer(cfg, model):
    if cfg.name == "Adam":
        return torch.optim.Adam(model.parameters(),
                                                              lr=cfg.base_lr,
                                                              betas=(cfg.beta1, cfg.beta2))                                                 
