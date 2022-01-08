import torch


def build_optimizer(cfg, model):
    if cfg.name == "Adam":
        return torch.optim.Adam([param for param in model.parameters() if param.requires_grad ],
                                                              lr=cfg.base_lr,
                                                              betas=(cfg.beta1, cfg.beta2))                                                 
