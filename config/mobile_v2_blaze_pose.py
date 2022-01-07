from easydict import EasyDict as edict

backbone_cfg=edict()
backbone_cfg.name="MobileNetV2"
backbone_cfg.widen_factor=1.0
backbone_cfg.resum_path=""
backbone_cfg.in_channels=3
backbone_cfg.out_indices=-1
backbone_cfg.frozen_stages=None
backbone_cfg.pre_name="mobilenet_v2"
backbone_cfg.base_path = "./pth"

###############################################################################################
head_cfg = edict()
head_cfg.name = "TopdownRegressionBlazeHead"
head_cfg.pre_name="blaze_head"
head_cfg.resum_path=""
head_cfg.base_path = "./pth"
head_cfg.with_mask_layers=False
head_cfg.joint_num=17
#########################################################################
loc_loss_cfg = edict()
loc_loss_cfg.name = "L1Loss"
loc_loss_cfg.use_target_weight = True
loc_loss_cfg.loss_weight = 1.
head_cfg.loc_loss = loc_loss_cfg
##################################################################################################
mask_loss_cfg = edict()
mask_loss_cfg.name = "VisMaskBCELoss"
mask_loss_cfg.use_target_weight = True
mask_loss_cfg.loss_weight = 1.0
head_cfg.mask_loss=mask_loss_cfg
##################################################################################################
data_train_cfg = edict()
data_train_cfg.name = "CocoDataRegression"
data_train_cfg.image_root = "/media/gpuser/bf9802a8-1070-44ed-8869-f0612c5c7f7c/lyc/datasets/humon_pose/coco/images/"
data_train_cfg.annotion_file = "/media/gpuser/bf9802a8-1070-44ed-8869-f0612c5c7f7c/lyc/datasets/humon_pose/coco/annotations/person_keypoints_train2017.json"
data_train_cfg.with_mask=False
data_train_cfg.image_size = [256, 256]          # 192
data_train_cfg.num_joints = 17
data_train_cfg.mean=[0.485, 0.456, 0.406]
data_train_cfg.std=[0.229, 0.224, 0.225]
data_train_cfg.batch_size = 64
data_train_cfg.pin_memory = False
data_train_cfg.num_workers = 4

data_train_cfg.is_rot = True
data_train_cfg.rot_factor = 40
data_train_cfg.rot_prob = 0.4

data_train_cfg.expan_factor=0.4                     # 最多外扩20%
data_train_cfg.expan_prob=0.6
data_train_cfg.min_expan_factor=0.2             # 最少外扩10%

data_train_cfg.is_shift = True              # 中心点会进行平移

data_train_cfg.is_pic = True
data_train_cfg.brightness_delta = 32
data_train_cfg.brightness_prob = 0.2
data_train_cfg.contrast_range = (0.5, 1.5)
data_train_cfg.contrast_prob = 0.2
data_train_cfg.saturation_range = (0.5, 1.5)
data_train_cfg.saturation_prob = 0.2
data_train_cfg.hue_delta = 18
data_train_cfg.hue_prob = 0.2
##################################################################################################
optimizer_cfg = edict()
optimizer_cfg.name = "Adam"
optimizer_cfg.base_lr = 5e-4
optimizer_cfg.weight_decay = 0
optimizer_cfg.beta1 = 0.9
optimizer_cfg.beta2 = 0.999
########################################################################################
lr_schedule_cfg = edict()
lr_schedule_cfg.name = "LrScheduleStep"
lr_schedule_cfg.step = [170, 200]
lr_schedule_cfg.gamma = 0.1
lr_schedule_cfg.min_lr = None
lr_schedule_cfg.warmup = "linear"
lr_schedule_cfg.warmup_iters = 500
lr_schedule_cfg.warmup_ratio = 0.001
########################################################
model_cfg = edict()
model_cfg.backbone = backbone_cfg
model_cfg.head = head_cfg
model_cfg.us_multi_gpus = False
model_cfg.gup_ids = [0, 1, 2, 3]
model_cfg.device = "cuda:3"

data_cfg = edict()
data_cfg.train = data_train_cfg

config = edict()
config.model = model_cfg
# config.loss_num=2
config.data = data_cfg
config.optimizer = optimizer_cfg
config.lr_schedule = lr_schedule_cfg

config.logfile = "./logger/blaze_logger.txt"
config.logger_freq=50
config.device = config.model.device
config.start_epoch = 0
config.end_epoch = 210
config.save_ckps_freq = 5