from easydict import EasyDict as edict


backbone_cfg = edict()
backbone_cfg.name = "RSN"
backbone_cfg.unit_channels=256
backbone_cfg.num_stages=1
backbone_cfg.num_units=4
backbone_cfg.num_blocks=[2, 2, 2, 2]            # 3,4,6,3
backbone_cfg.num_steps=4
backbone_cfg.res_top_channels=64
backbone_cfg.expand_times=26
###############################################################################################
head_cfg = edict()
head_cfg.name = "TopdownHeatmapMSMUHead"
head_cfg.out_shape=[64, 48]
head_cfg.unit_channels=256
head_cfg.out_channels=17
head_cfg.num_stages=1
head_cfg.num_units=4
head_cfg.use_prm=False

loss_cfg_01 = edict()
loss_cfg_01.name = "JointsMSELoss"
loss_cfg_01.use_target_weight = True
loss_cfg_01.loss_weight = 0.25

loss_cfg_02 = edict()
loss_cfg_02.name = "JointsMSELoss"
loss_cfg_02.use_target_weight = True
loss_cfg_02.loss_weight = 0.25

loss_cfg_03 = edict()
loss_cfg_03.name = "JointsMSELoss"
loss_cfg_03.use_target_weight = True
loss_cfg_03.loss_weight = 0.25

loss_cfg_04 = edict()
loss_cfg_04.name = "JointsOHKMMSELoss"
loss_cfg_04.use_target_weight = True
loss_cfg_04.loss_weight = 1.

head_cfg.loss = [loss_cfg_01, loss_cfg_02, loss_cfg_03, loss_cfg_04]

##################################################################################################
data_train_cfg = edict()
data_train_cfg.name = "CocoDataWithMutex"
data_train_cfg.is_train = True
data_train_cfg.image_root = "/home/lyc/human_pose/coco/images/"
data_train_cfg.annotion_file = "/home/lyc/human_pose/coco/annotations/person_keypoints_train2017.json"
data_train_cfg.image_size = [192, 256]
data_train_cfg.num_joints = 17
data_train_cfg.mean=[0.485, 0.456, 0.406]
data_train_cfg.std=[0.229, 0.224, 0.225]
data_train_cfg.batch_size = 96
data_train_cfg.pin_memory = False
data_train_cfg.num_workers = 4

heatmaps_01 = edict()
heatmaps_01.name = "MegviiHeatmap"
heatmaps_01.kernel = [11, 11]
heatmaps_01.num_joints = data_train_cfg.num_joints
heatmaps_01.image_size = data_train_cfg.image_size
heatmaps_01.heatmap_size =  [data_train_cfg.image_size[0] // 4, data_train_cfg.image_size[1] // 4]
heatmaps_01.joint_weights = None
heatmaps_01.use_different_joint_weights = False

heatmaps_02 = edict()
heatmaps_02.name = "MegviiHeatmap"
heatmaps_02.kernel = [9, 9]
heatmaps_02.num_joints = data_train_cfg.num_joints
heatmaps_02.image_size = data_train_cfg.image_size
heatmaps_02.heatmap_size =  [data_train_cfg.image_size[0] // 4, data_train_cfg.image_size[1] // 4]
heatmaps_02.joint_weights = None
heatmaps_02.use_different_joint_weights = False

heatmaps_03 = edict()
heatmaps_03.name = "MegviiHeatmap"
heatmaps_03.kernel = [7, 7]
heatmaps_03.num_joints = data_train_cfg.num_joints
heatmaps_03.image_size = data_train_cfg.image_size
heatmaps_03.heatmap_size =  [data_train_cfg.image_size[0] // 4, data_train_cfg.image_size[1] // 4]
heatmaps_03.joint_weights = None
heatmaps_03.use_different_joint_weights = False

heatmaps_04 = edict()
heatmaps_04.name = "MegviiHeatmap"
heatmaps_04.kernel = [5, 5]
heatmaps_04.num_joints = data_train_cfg.num_joints
heatmaps_04.image_size = data_train_cfg.image_size
heatmaps_04.heatmap_size =  [data_train_cfg.image_size[0] // 4, data_train_cfg.image_size[1] // 4]
heatmaps_04.joint_weights = None
heatmaps_04.use_different_joint_weights = False

data_train_cfg.heatmaps = [heatmaps_01, heatmaps_02, heatmaps_03, heatmaps_04]

data_train_cfg.is_flip = True
data_train_cfg.flip_prob = 0.5
data_train_cfg.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

data_train_cfg.is_half_body = True
data_train_cfg.num_joints_half_body = 8
data_train_cfg.prob_half_body = 0.3
data_train_cfg.upper_body_index =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data_train_cfg.lower_body_index = [11, 12, 13, 14, 15, 16]

data_train_cfg.is_rot = True
data_train_cfg.rot_factor = 40
data_train_cfg.scale_factor = 0.25
data_train_cfg.rot_prob = 0.6

data_train_cfg.is_pic = True
data_train_cfg.brightness_delta = 32
data_train_cfg.contrast_range = (0.5, 1.5)
data_train_cfg.saturation_range = (0.5, 1.5)
data_train_cfg.hue_delta = 18
##################################################################################################
data_eval_cfg = edict()
data_eval_cfg.name = "CocoDataEval"
data_eval_cfg.is_train = False
data_eval_cfg.image_root = "/home/lyc/human_pose/coco/images/"
data_eval_cfg.annotion_file = "/home/lyc/human_pose/coco/annotations/person_keypoints_val2017.json"
data_eval_cfg.image_size = [192, 256]
data_eval_cfg.num_joints = 17
data_eval_cfg.mean=[0.485, 0.456, 0.406]
data_eval_cfg.std=[0.229, 0.224, 0.225]
data_eval_cfg.batch_size = 32
data_eval_cfg.pin_memory = False
data_eval_cfg.num_workers = 4
##################################################################################################
optimizer_cfg = edict()
optimizer_cfg.name = "Adam"
optimizer_cfg.base_lr = 2e-2
optimizer_cfg.weight_decay = 0
optimizer_cfg.beta1 = 0.9
optimizer_cfg.beta2 = 0.999
########################################################################################
lr_schedule_cfg = edict()
lr_schedule_cfg.name = "LrScheduleStep"
lr_schedule_cfg.step = [170, 190, 200]
lr_schedule_cfg.gamma = 0.1
lr_schedule_cfg.min_lr = None
lr_schedule_cfg.warmup = "linear"
lr_schedule_cfg.warmup_iters = 500
lr_schedule_cfg.warmup_ratio = 0.001
########################################################
model_cfg = edict()
model_cfg.backbone = backbone_cfg
model_cfg.head = head_cfg
model_cfg.resum_path = "./pth/rsn_net.pth"
model_cfg.us_multi_gpus = True
model_cfg.gup_ids = [0, 1]
model_cfg.device = "cuda:0"

data_cfg = edict()
data_cfg.train = data_train_cfg
data_cfg.eval = data_eval_cfg

config = edict()
config.model = model_cfg
config.data = data_cfg
config.optimizer = optimizer_cfg
config.lr_schedule = lr_schedule_cfg

eval_cfg = edict()
eval_cfg.name = "TopDownHumanPoseEval"
eval_cfg.data = data_eval_cfg
eval_cfg.device = model_cfg.device
config.eval = eval_cfg

config.logfile = "./logger/rsn_net.txt"
config.logger_freq=100
config.model_pth_name = "rsn_net.pth"
config.mode_root = "./pth"
config.device = config.model.device
config.start_epoch = 0
config.end_epoch = 210
config.eval_freq = 10
config.save_ckps_freq = 1

config.loss_num = len(config.model.head.loss ) + 1
