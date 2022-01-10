import torch
from tqdm import tqdm
from utils.utils import AverageMeter
from trainer.base import TrainBase


class SimpleTrainer(TrainBase):

    def __init__(self, model, data_loader, optimizer, device, lr_schedule, logger, logger_freq, update_loss):
        super().__init__(model, data_loader, optimizer,  device, lr_schedule, logger, logger_freq, update_loss)

    def train_one_epoch(self, epoch_index, save_ckps):
        self.model.train()
        loss_avgs = {}
        batch = 0
        for images, targets, target_weights in tqdm(iter(self.data_loader)):
                batch += 1
                self.lr_schedule.update_lr(epoch_index, batch)
                if type(images) == torch.Tensor:
                    images = images.to(self.device).type(torch.float)
                else:
                    images = [image.to(self.device).type(torch.float) for image in images]
                
                if type(targets) == torch.Tensor:
                    targets =targets.to(self.device).type(torch.float)
                else:
                    targets = [target.to(self.device).type(torch.float) for target in targets]

                if type(target_weights) == torch.Tensor:
                    target_weights = target_weights.to(self.device).type(torch.float)
                else:
                    target_weights = [target_weight.to(self.device).type(torch.float) for target_weight in target_weights]

                outputs = self.model(images)
                loss_inputs = outputs, targets, target_weights
                loss = self.get_loss(loss_inputs)

                self.optimizer.zero_grad()
                loss.get(self.update_loss).backward()
                self.optimizer.step()
        
                for key in loss.keys():
                    if key not in loss_avgs.keys():
                        loss_avgs[key] = AverageMeter()
                    loss_avgs[key].update(loss[key].data.item(), images.size(0))

                if batch % self.logger_freq == 0:
                    str_infos = "Epoch{}/Batch {}\tLR {:.6f}".format(epoch_index + 1, batch,
                                                                                                                     self.optimizer.param_groups[0]['lr'])
                    for key in loss_avgs.keys():
                        str_infos += "\t"
                        str_infos += key + " {loss_avg.val:.6f}({loss_avg.avg:.6f})".format(loss_avg=loss_avgs[key])
                    self.logger.info(str_infos)

        if save_ckps:
            self.save_ckps(epoch_index)
    