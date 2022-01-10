import torch


def build_optimizer(cfg, model):
    if cfg.name == "Adam":
        return torch.optim.Adam([param for param in model.parameters() if param.requires_grad ],
                                                              lr=cfg.base_lr,
                                                              betas=(cfg.get("beta1", 0.9), cfg.get("beta2", 0.99)),
                                                              weight_decay=cfg.get("weight_decay", 0))                                                 
