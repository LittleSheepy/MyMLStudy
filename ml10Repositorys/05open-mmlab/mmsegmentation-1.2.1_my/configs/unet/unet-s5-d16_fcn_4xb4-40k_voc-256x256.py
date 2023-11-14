_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)

data_root = r'F:\sheepy\02data\VOCdevkit\VOC2012/'
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
