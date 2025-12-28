import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='imgs to tfrecord')
    parser.add_argument('--data_path', default=None, type=str, required=True,
                        help='path to the parent directory of source image data')
    parser.add_argument('--src_name', default=None, type=str, required=True,
                        help='name of source image dataset')
    parser.add_argument('--path_suffix', default='images', type=str, required=True,
                        help='suffix of source image directory')

    # # FRCSyn, mix two-stage synthesized images
    # parser.add_argument('--augment_num', default=0, type=str, required=False,
    #                     help='number of augment images per identity')
    # parser.add_argument('--augment_path', default=None, type=str, required=False,
    #                     help='path to augment image directory')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.data_path is not None and args.src_name is not None

    output_file = os.path.join(args.data_path, args.src_name, f'{args.src_name}_img_list.txt')

    path = os.path.join(args.data_path, args.src_name, args.path_suffix)
    print('Path: {}'.format(path))

    # 获取所有子文件夹并排序
    subfolders = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))], key=int)

    print(output_file)
    # 打开输出文件
    with open(output_file, 'w') as f:
        # 遍历排序后的子文件夹
        index = 0
        for label in subfolders:
            label_path = os.path.join(path, label)
            print('Label: {} | Index: {}'.format(label, index))

            # 确保是一个目录
            if os.path.isdir(label_path):
                # 遍历子文件夹下的文件
                files = sorted([file_name for file_name in os.listdir(label_path)
                                if file_name.lower().endswith(('.jpg', '.jpeg', '.png'))],
                               key=lambda x: int(os.path.splitext(x)[0]))

                for file_name in files:
                    # 获取文件的相对路径
                    relative_path = os.path.join(args.path_suffix, label, file_name)
                    # 写入 b.txt 文件，每行格式为 "相对路径 label"
                    f.write(f"{relative_path}\t{index}\n")
                    f.flush()

                # if args.augment_num > 0:
                #     assert args.augment_path is not None
                #     for i in range(args.augment_num):
                #         f.write("")
                index += 1

    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
