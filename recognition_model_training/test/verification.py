import os
import sys
import argparse
import numpy as np
import torch
from utils import perform_val_bin

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from torchkit.backbone import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='verfication tool')
    parser.add_argument('--ckpt_path', default=None, required=True, help='model_path')
    parser.add_argument('--backbone', default='IR_50', help='backbone type')
    parser.add_argument('--gpu_ids', default='0', help='gpu ids')
    parser.add_argument('--batch_size', default=128, help='batch size')
    parser.add_argument('--data_root', default='', required=True, help='validation data root')
    parser.add_argument('--embedding_size', default=512, help='embedding_size')
    parser.add_argument('--epoch', default=24, help='checkpoint epoch')
    # parser.add_argument('--use_latent', action='store_true', help='use latent input (with upsample layers)')
    args = parser.parse_args()
    return args


def get_val_pair_from_bin(path, name):
    """ read data for bin files
    """
    import pickle
    import cv2
    path = os.path.join(path, name)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    images = []
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        # image = Image.frombytes('RGB', (112, 112), _bin, 'raw')
        # image = mx.image.imdecode(_bin)
        image = np.frombuffer(_bin, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.transpose(image, axes=(2, 0, 1))
        img = img.astype(np.float32)
        img = (img / 255. - 0.5) / 0.5
        images.append(img)
    issame_list = np.array(issame_list)
    print(len(images))
    return images, issame_list


def get_val_data_from_bin(data_path):
    lfw, lfw_issame = get_val_pair_from_bin(data_path, 'lfw.bin')
    cfp_fp, cfp_fp_issame = get_val_pair_from_bin(data_path, 'cfp_fp.bin')
    agedb_30, agedb_30_issame = get_val_pair_from_bin(data_path, 'agedb_30.bin')
    calfw, calfw_issame = get_val_pair_from_bin(data_path, 'calfw.bin')
    cplfw, cplfw_issame = get_val_pair_from_bin(data_path, 'cplfw.bin')
    return (lfw, cfp_fp, agedb_30, cplfw, calfw,
            lfw_issame, cfp_fp_issame, agedb_30_issame,
            cplfw_issame, calfw_issame)


def get_val_pair(path, name):
    """ read data from boclz dir
    """
    import bcolz
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')

    return (lfw, cfp_fp, agedb_30, cplfw, calfw,
            lfw_issame, cfp_fp_issame, agedb_30_issame,
            cplfw_issame, calfw_issame)


def main():
    """ Perform evaluation on LFW, CFP-FP, AgeDB, CALFW, CPLFW datasets,
        each dataset consists of some positive and negative pair data.
    """
    args = parse_args()
    torch.manual_seed(1337)
    input_size = [112, 112]
    # load backbone
    
    backbone = get_model(args.backbone)(input_size)
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError("%s not exists" % args.ckpt_path)
    backbone.load_state_dict(torch.load(args.ckpt_path))
    
    #backbone_model = get_model(args.backbone)
    #original_backbone = backbone_model(input_size)
    # 检查是否需要添加latent输入层（用于训练时使用latent的模型）
    #if args.latent_input:
    #    backbone = torch.nn.Sequential(
    #        torch.nn.ConvTranspose2d(3, 3, kernel_size=4, stride=4),  # 32x32 → 128x128
    #        torch.nn.Upsample(size=(112, 112), mode='bilinear', align_corners=False),  # 裁成112x112
    #        original_backbone
    #    )
    #else:
    #    backbone = original_backbone
        
    #if not os.path.exists(args.ckpt_path):
    #    raise RuntimeError("%s not exists" % args.ckpt_path)
    #backbone.load_state_dict(torch.load(args.ckpt_path))



    val_data_dir = args.data_root
    # load data
    lfw, cfp_fp, agedb_30, cplfw, calfw, \
        lfw_issame, cfp_fp_issame, agedb_30_issame, \
        cplfw_issame, calfw_issame = get_val_data_from_bin(val_data_dir)

    # backbone to gpu
    gpus = [int(x) for x in args.gpu_ids.rstrip().split(',')]
    if len(gpus) > 1:
        backbone = torch.nn.DataParallel(backbone, device_ids=gpus)
        backbone = backbone.cuda()
    else:
        backbone = backbone.cuda()

    print("Perform Evaluation on LFW, CFP_FP, AgeDB, CPLFW...")
    # LFW result
    accuracy_lfw, best_threshold_lfw, bad_case_lfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        lfw,
        lfw_issame)
    print("Evaluation: LFW Acc: {}, thresh: {}".format(accuracy_lfw, best_threshold_lfw))
    # CFP-FP result
    accuracy_cfp_fp, best_threshold_cfp_fp, bad_case_cfp_fp = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        cfp_fp,
        cfp_fp_issame)
    # AgeDB result
    print("Evaluation: CFP_FP Acc: {}, thresh: {}".format(accuracy_cfp_fp, best_threshold_cfp_fp))
    accuracy_agedb, best_threshold_agedb, bad_case_agedb = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        agedb_30,
        agedb_30_issame)
    # CALFW result
    print("Evaluation: AgeDB Acc: {}, thresh: {}".format(accuracy_agedb, best_threshold_agedb))
    accuracy_cplfw, best_threshold_cplfw, bad_case_cplfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        cplfw,
        cplfw_issame)
    print("Evaluation: CPLFW Acc: {}, thresh: {}".format(accuracy_cplfw, best_threshold_cplfw))
    accuracy_calfw, best_threshold_calfw, bad_case_calfw = perform_val_bin(
        args.embedding_size,
        args.batch_size,
        backbone,
        calfw,
        calfw_issame)
    # CPLFW result
    print("Evaluation: CALFW Acc: {}, thresh: {}".format(accuracy_calfw, best_threshold_calfw))

    acc_results = [
        round(accuracy_lfw * 100, 2),
        round(accuracy_cfp_fp * 100, 2),
        round(accuracy_agedb * 100, 2),
        round(accuracy_cplfw * 100, 2),
        round(accuracy_calfw * 100, 2)
    ]

    avg = round(sum(acc_results) / len(acc_results), 2)

    print('Results: {}\t{}\t{}\t{}\t{}'.format(*acc_results))
    print('Average results: {}'.format(avg))

    # assume checkpoint path is "/a/b/ckpt/ckpt.pth", experiment path is "a/b"
    file_path = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), f"test/result_ep{str(args.epoch)}.txt")
    print(file_path) 
    if os.path.isfile(file_path):
        with open(file_path, 'a') as file:
            file.write(f"LFW\t{round(accuracy_lfw * 100, 2)}\n")
            file.write(f"CFP-FP\t{round(accuracy_cfp_fp * 100, 2)}\n")
            file.write(f"AgeDB\t{round(accuracy_agedb * 100, 2)}\n")
            file.write(f"CPLFW\t{round(accuracy_cplfw * 100, 2)}\n")
            file.write(f"CALFW\t{round(accuracy_calfw * 100, 2)}\n")

    save_bad_case = True
    if save_bad_case == True:
        bad_case_file_path = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)),
                                          f'test/bad_case_ep{str(args.epoch)}.txt')
        with open(bad_case_file_path, 'a') as f:
            f.write('LFW ' + ' '.join(map(str, bad_case_lfw)) + "\n")
            f.write('CFP-FP ' + ' '.join(map(str, bad_case_cfp_fp)) + "\n")
            f.write('AgeDB ' + ' '.join(map(str, bad_case_agedb)) + "\n")
            f.write('CPLFW ' + ' '.join(map(str, bad_case_cplfw)) + "\n")
            f.write('CALFW ' + ' '.join(map(str, bad_case_calfw)) + "\n")


if __name__ == "__main__":
    main()
