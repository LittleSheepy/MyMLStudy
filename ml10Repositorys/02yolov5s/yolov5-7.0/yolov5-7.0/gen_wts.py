import sys
import argparse
import os
import struct
import torch
# from utils.torch_utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='Convert .pt file to .wts')
    parser.add_argument('-w', '--weights', default=pt_path, help='Input weights (.pt) file path (required)')
    parser.add_argument('-o', '--output',default=wts_path,help='Output (.wts) file path (optional)')
    parser.add_argument(
        '-t', '--type', type=str, default='seg', choices=['detect', 'cls', 'seg'],
        help='determines the model is detection/classification')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid input file')
    if not args.output:
        args.output = os.path.splitext(args.weights)[0] + '.wts'
    elif os.path.isdir(args.output):
        args.output = os.path.join(
            args.output,
            os.path.splitext(os.path.basename(args.weights))[0] + '.wts')
    return args.weights, args.output, args.type


def generate_wts(pt_file, wts_file, m_type = "cls"):
    print(f'Generating .wts for {m_type} model')
    # Load model
    print(f'Loading {pt_file}')
    device = torch.device('cuda:0')
    model = torch.load(pt_file, map_location=device)  # Load FP32 weights
    model = model['ema' if model.get('ema') else 'model'].float()

    if m_type in ['detect', 'seg']:
        # update anchor_grid info
        anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
        # model.model[-1].anchor_grid = anchor_grid
        delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer
        # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
        model.model[-1].register_buffer("anchor_grid", anchor_grid)
        model.model[-1].register_buffer("strides", model.model[-1].stride)

    model.to(device).eval()

    print(f'Writing into {wts_file}')
    with open(wts_file, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')


if __name__ == '__main__':
    root_dir = r"F:\sheepy\00MyMLStudy\ml10Repositorys\02yolov5s\yolov5-7.0\yolov5-7.0\runs\16_Front_YW\20231009\weights\/"
    pt_path = root_dir + "best.pt"
    # wts_path = root_dir + "KDAI_01_Front_BlackGray_1024_1024_1_NMLZPS_0.6_1017_pre.wts"
    # wts_path = root_dir + "KDAI_02_Front_BlackGray_1024_1024_1_NMNICPS_0.3_091320.wts"
    # wts_path = root_dir + "KDAI_03_Front_BlackGray_1024_1024_1_NMJQJPS_0.6_0913.wts"
    # wts_path = root_dir + "KDAI_04_Front_BlackGray_1024_1024_1_NMLBPS_0.6_0829.wts"
    # wts_path = root_dir + "KDAI_05_Front_BlackGray_1024_1024_1_NMJY_0.3_0914.wts"
    # wts_path = root_dir + "KDAI_06_Front_BlackGray_1024_1024_1_NMSZ_0.6_1017_pre.wts"
    # wts_path = root_dir + "KDAI_07_Front_BlackGray_1024_1024_1_NMWR_0.6_0911.wts"

    #wts_path = root_dir + "KDAI_12_Side_BlackGray_1024_1024_1_CMPS_0.6_1017_pre.wts"
    wts_path = root_dir + "KDAI_16_Side_BlackGray_1024_1024_1_CMYW_0.6_1017.wts"
    # wts_path = root_dir + "KDAI_14_Side_BlackGray_1024_1024_1_CMWR_0.6_1017.wts"
    #wts_path = root_dir + "seg_best_nmjqjps.wts"
    pt_file, wts_file, m_type = parse_args()
    generate_wts(pt_file, wts_file, m_type)