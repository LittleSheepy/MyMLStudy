_base_ = [
    './deeplabv3plus_r50-d8.py',
    './pascal_voc12.py',
    './default_runtime.py',
    './schedule_20k.py'
]
model = dict(
    decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
