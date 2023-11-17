import os
from mmseg.apis import inference_model, init_model, show_result_pyplot

# Specify the path to model config and checkpoint file
config_path = r"configs/mae/mae-base_upernet_8xb2-amp-160k_ade20k-512x512.py"
work_dir = r"E:\01model_output\01mmseg/mae-base_upernet_8xb2-amp-160k_ade20k-512x512_train512_01"
config_file = config_path
checkpoint_file = r"E:\01model_output\01mmseg\mae-base_upernet_8xb2-amp-160k_ade20k-512x512_train512_01\/iter_120000.pth"

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')

# test a directory of images
img_dir = r'E:\0ProjectData\AI_PK_pictreue\1_Front_LZPS\all\/'
# img_dir = r'E:\0ProjectData\AI_PK_pictreue\1_Front_LZPS\img\/'
imgs = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith('.jpg')]

img_cnt = 0
def save_logit(i_seg_logits):
    import cv2
    import torch.nn.functional as F
    global img_cnt
    i_seg_logits_sig = i_seg_logits.data.sigmoid()
    i_seg_logits_soft = output = F.softmax(i_seg_logits.data, dim=0)
    i_seg_logits_sig1 = i_seg_logits_sig[1, :, :]
    i_seg_logits_sig1 = i_seg_logits_sig1.data.cpu().numpy()*200
    img_path = r"F:\sheepy\00MyMLStudy\ml10Repositorys\05open-mmlab\mmsegmentation-1.2.1_my\output/" + str(img_cnt) + ".jpg"
    cv2.imwrite(img_path, i_seg_logits_sig1)
    img_cnt = img_cnt + 1
img_cnt1 = 0
for img in imgs:
    for i, result in enumerate(inference_model(model, [img])):
        # show_result_pyplot(model, imgs[i], result)
        output_file = os.path.join('output', f'result_{img_cnt1}.png')

        save_logit(result.seg_logits)
        vis_iamge = show_result_pyplot(model, img, result, out_file=output_file, show=False)
        # save the result
        # model.show_result(imgs[i], result, out_file=output_file)
        img_cnt1 = img_cnt1 + 1


