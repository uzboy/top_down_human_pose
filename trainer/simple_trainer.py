import torch
from tqdm import tqdm
from utils.utils import AverageMeter
from trainer.base import TrainBase


class SimpleTrainer(TrainBase):

    def __init__(self, model, data_loader, optimizer, device, lr_schedule, logger, logger_freq):
        super().__init__(model, data_loader, optimizer,  device, lr_schedule, logger, logger_freq)

    def train_one_epoch(self, epoch_index, save_ckps):
        self.model.train()
        loss_avgs = []
        batch = 0
        for images, targets, target_weights in tqdm(iter(self.data_loader)):
                batch += 1
                self.lr_schedule.update_lr(epoch_index, batch)
                images = images.to(self.device).type(torch.float)
                targets =targets.to(self.device).type(torch.float)
                target_weights = target_weights.to(self.device).type(torch.float)

                outputs = self.model(images)
                loss_inputs = outputs, targets, target_weights
                loss = self.get_loss(loss_inputs)

                if not isinstance(loss, list):
                    loss = [loss]

                self.optimizer.zero_grad()
                loss[-1].backward()
                self.optimizer.step()
        
                for index in range(self.loss_nums):
                    if len(loss_avgs) < (index + 1):
                        loss_avgs.append(AverageMeter())

                    loss_avgs[index].update(loss[index].data.item(), images.size(0))

                if batch % self.logger_freq == 0:
                    str_infos = "Epoch{}/Batch {}\tLR {:.6f}\t".format(epoch_index + 1, batch,
                                                                                                                        self.optimizer.param_groups[0]['lr'])
                    for index in range(self.loss_nums):
                        if index == self.loss_nums - 1:
                            str_infos += "total_loss {loss_avg.val:.6f}({loss_avg.avg:.6f})".format(loss_avg=loss_avgs[index])
                        else:
                            str_infos += "loss_{} {loss_avg.val:.6f}({loss_avg.avg:.6f})".format(index + 1, loss_avg=loss_avgs[index])
                        
                        if index != self.loss_nums - 1:
                            str_infos += "\t"
    
                    self.logger.info(str_infos)

        if save_ckps:
            self.save_ckps(epoch_index)