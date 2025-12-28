import os
import cv2
import argparse
import torch
import numpy as np
import tensorflow as tf
from skimage import transform as trans
from datetime import datetime as dt


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='imgs to tfrecord')
    parser.add_argument('--src_pth_dir', default=None, type=str, required=True,
                        help='path to source latent dictionaries')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='path to output meta and tfrecords')
    parser.add_argument('--tfrecords_name', default='TFR-Latent-CASIA', type=str, required=True,
                        help='name of output tfrecord')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tfrecords_dir = os.path.join(args.output_dir, args.tfrecords_name)
    tfrecords_name = args.tfrecords_name
    if not os.path.isdir(tfrecords_dir):
        os.makedirs(tfrecords_dir)
        os.makedirs(os.path.join(tfrecords_dir, tfrecords_name))

    count = 0
    cur_shard_size = 0
    cur_shard_idx = -1
    cur_shard_writer = None
    cur_shard_path = None
    cur_shard_offset = None
    idx_writer = open(os.path.join(tfrecords_dir, "%s.txt" % tfrecords_name), 'w')

    class_id_map = {}
    next_class_idx = 0

    for pth_file in sorted(os.listdir(args.src_pth_dir)):
        if not pth_file.endswith(".pth"):
            continue
        pth_path = os.path.join(args.src_pth_dir, pth_file)
        data_dict = torch.load(pth_path, map_location='cpu')

        for key, tensor in data_dict.items():
            raw_class_id = key.split('/')[0]
            if raw_class_id not in class_id_map:
                class_id_map[raw_class_id] = next_class_idx
                next_class_idx += 1
            class_id = class_id_map[raw_class_id]

            tensor_bytes = tensor.numpy().tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'tensor': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_bytes])),
            }))

            if cur_shard_size == 0:
                print("{}: {} processed".format(dt.now(), count))
                cur_shard_idx += 1
                record_filename = '{0}-{1:05}.tfrecord'.format(tfrecords_name, cur_shard_idx)
                if cur_shard_writer is not None:
                    cur_shard_writer.close()
                cur_shard_path = os.path.join(tfrecords_dir, tfrecords_name, record_filename)
                cur_shard_writer = tf.io.TFRecordWriter(cur_shard_path)
                cur_shard_offset = 0

            example_bytes = example.SerializeToString()
            cur_shard_writer.write(example_bytes)
            cur_shard_writer.flush()
            idx_writer.write(f'{tfrecords_name}\t{cur_shard_idx}\t{cur_shard_offset}\t{class_id}\n')
            cur_shard_offset += (len(example_bytes) + 16)

            if cur_shard_size % 5000 == 0:
                print(f"{cur_shard_size} / 500000")

            count += 1
            cur_shard_size = (cur_shard_size + 1) % 500000

    if cur_shard_writer is not None:
        cur_shard_writer.close()
    idx_writer.close()
    print(f'Total examples = {count}')
    print(f'Total shards = {cur_shard_idx + 1}')

    # with open(args.img_list, 'r') as f:
    #     for line in f:
    #         img_path, label = line.split('\t')[0], line.split('\t')[1]
    #         # print(label)
    #         img = cv2.imread(os.path.join(args.src_pth_dir, img_path))
    #         img_bytes = cv2.imencode('.jpg', img)[1].tostring()

    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
    #         }))

    #         if cur_shard_size == 0:
    #             print("{}: {} processed".format(dt.now(), count))
    #             cur_shard_idx += 1
    #             record_filename = '{0}-{1:05}.tfrecord'.format(tfrecords_name, cur_shard_idx)
    #             if cur_shard_writer is not None:
    #                 cur_shard_writer.close()
    #             cur_shard_path = os.path.join(tfrecords_dir, tfrecords_name, record_filename)  # 嵌套一层
    #             cur_shard_writer = tf.io.TFRecordWriter(cur_shard_path)
    #             cur_shard_offset = 0

    #         example_bytes = example.SerializeToString()
    #         cur_shard_writer.write(example_bytes)
    #         cur_shard_writer.flush()
    #         # idx_writer.write('{}\t{}\t{}\n'.format(img_path, cur_shard_idx, cur_shard_offset))
    #         idx_writer.write('{}\t{}\t{}\t{}'.format(tfrecords_name, cur_shard_idx, cur_shard_offset, label))
    #         cur_shard_offset += (len(example_bytes) + 16)

    #         if cur_shard_size % 5000 == 0:
    #             print("%d / %d" % (cur_shard_size, 500000))

    #         count += 1
    #         cur_shard_size = (cur_shard_size + 1) % 500000

    # if cur_shard_writer is not None:
    #     cur_shard_writer.close()
    # idx_writer.close()
    # print('total examples number = {}'.format(count))
    # print('total shard number = {}'.format(cur_shard_idx + 1))


if __name__ == '__main__':
    main()
