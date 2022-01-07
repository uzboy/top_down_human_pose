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
head_cfg.name = "TopdownHeatmapSimpleHead"
head_cfg.pre_name="mobilenet_v2_top_simple_head"
head_cfg.resum_path=""
head_cfg.base_path = "./pth"
head_cfg.in_channels = 1280
head_cfg.out_channels = 17
head_cfg.num_deconv_layers = 3
head_cfg.num_deconv_filters=(256, 256, 256)
head_cfg.num_deconv_kernels=(4, 4, 4)
head_cfg.num_conv_layers=0
head_cfg.conv_layers_out=None
head_cfg.conv_layer_kernel=None
head_cfg.in_index=0
head_cfg.input_transform=None
head_cfg.align_corners=False
head_cfg.use_sigmoid = True
#########################################################################
loss_cfg = edict()
loss_cfg.name = "JointsBCELoss"
loss_cfg.use_target_weight = True
loss_cfg.loss_weight = 1.
head_cfg.loss = loss_cfg
##################################################################################################
data_train_cfg = edict()
data_train_cfg.name = "CocoDataWithBox"
data_train_cfg.image_root = "/media/gpuser/bf9802a8-1070-44ed-8869-f0612c5c7f7c/lyc/datasets/humon_pose/coco/images/"
data_train_cfg.annotion_file = "/media/gpuser/bf9802a8-1070-44ed-8869-f0612c5c7f7c/lyc/datasets/humon_pose/coco/annotations/person_keypoints_train2017.json"
data_train_cfg.image_size = [256, 256]          # 192
data_train_cfg.num_joints = 17
data_train_cfg.mean=[0.485, 0.456, 0.406]
data_train_cfg.std=[0.229, 0.224, 0.225]
data_train_cfg.batch_size = 64
data_train_cfg.pin_memory = False
data_train_cfg.num_workers = 4

heatmaps_01 = edict()
heatmaps_01.name = "MSRAHeatmap"
heatmaps_01.sigma   = 2
heatmaps_01.unbiased_encoding = False
heatmaps_01.num_joints = data_train_cfg.num_joints
heatmaps_01.image_size = data_train_cfg.image_size
heatmaps_01.heatmap_size =  [data_train_cfg.image_size[0] // 4, data_train_cfg.image_size[1] // 4]
heatmaps_01.joint_weights = None
heatmaps_01.use_different_joint_weights = False

data_train_cfg.heatmaps = [heatmaps_01]

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
config.data = data_cfg
config.optimizer = optimizer_cfg
config.lr_schedule = lr_schedule_cfg

config.logfile = "./logger/mobilenet_v2.txt"
config.logger_freq=50
config.device = config.model.device
config.start_epoch = 0
config.end_epoch = 210
config.save_ckps_freq = 5
