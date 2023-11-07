import mmseg
from mmseg.apis import MMSegInferencer

def inference():
    # 将模型加载到内存中
    inferencer = MMSegInferencer(model='deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024')
    # 推理
    inferencer('demo/demo.png', show=True)

if __name__ == '__main__':
    print("")
    print(mmseg.__version__)