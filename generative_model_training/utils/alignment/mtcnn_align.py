import os
import cv2
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from torchvision.utils import save_image

from arcface import norm_crop

from facenet_pytorch import MTCNN

from PIL import Image

import sys

sys.path.insert(0, '/IDiff-Face//')

# select_largest=True：如果检测到多张人脸，只选择最大的那张；min_face_size=1：最小人脸尺寸为1像素（实际会检测所有尺寸的人脸）；post_process=False：不进行后处理（如标准化），保持原始检测结果
mtcnn = MTCNN(
    select_largest=True, min_face_size=1, post_process=False, device="cuda:0"
)


def load_image_paths(datadir, num_imgs=0):
    """load num_imgs many FFHQ images"""
    img_files = sorted(os.listdir(datadir))
    if num_imgs != 0:
        img_files = img_files[:num_imgs]
    return [os.path.join(datadir, f_name) for f_name in img_files]


class ImageInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, num_imgs=0):

        """Initializes image paths and preprocessing module."""
        self.img_paths = load_image_paths(datadir, num_imgs)
        print("Number of images:", len(self.img_paths))
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = Image.open(self.img_paths[index])
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, os.path.basename(self.img_paths[index])

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)


def detect_and_align_single_batch(img_batch, img_names,mtcnn,out_folder,evalDB=False):
    img_batch = img_batch.permute(0, 2, 3, 1)
    img_batch = (img_batch * 255)

    _, _, landmarks = mtcnn.detect(img_batch, landmarks=True) # 检查人脸和关键点，返回边界框坐标、置信度分数和关键点坐标

    img_batch = img_batch.detach().cpu().numpy()

    skipped_imgs=[]
    landmarks_list=[]
    for img, img_name, landmark in zip(img_batch, img_names, landmarks):
        if landmark is None:
            skipped_imgs.append(img_name)
            continue

        facial5points = landmark[0]
        landmarks_list.append(facial5points)
        warped_face = norm_crop(img, landmark=facial5points, image_size=112, createEvalDB=evalDB)

        save_image(torch.from_numpy(warped_face).permute(2, 0, 1) / 255.0, os.path.join(out_folder, img_name))
        counter += 1
    return skipped_imgs,counter,landmarks_list

def align_images(in_folder, out_folder, batch_size, evalDB=False):
    """MTCNN alignment for all images in in_folder and save to out_folder
    args:
            in_folder: folder path with images
            out_folder: where to save the aligned images
            batch_size: batch size
            num_imgs: amount of images to align - 0: align all images
            evalDB: evaluation DB alignment
    """
    os.makedirs(out_folder, exist_ok=True)

    dataset = ImageInferenceDataset(datadir=in_folder)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2
    )

    skipped_imgs = []
    counter = 0
    landmark_statistics = []

    for img_batch, img_names in tqdm(dataloader):
        img_batch = img_batch.to("cuda:0")\

        img_batch = img_batch.permute(0, 2, 3, 1) # (B,C,H,W)->(B,H,W,C)
        img_batch = (img_batch * 255)

        _, _, landmarks = mtcnn.detect(img_batch, landmarks=True)

        img_batch = img_batch.detach().cpu().numpy()

        for img, img_name, landmark in zip(img_batch, img_names, landmarks):
            if landmark is None:
                skipped_imgs.append(img_name)
                continue
            facial5points = landmark[0] #提取第一张人脸的5个关键点为什么[0]：MTCNN可能检测到多张人脸，这里取第一张（通常是最大的）形状：(5, 2)

            landmark_statistics.append(facial5points)
            warped_face = norm_crop(img, landmark=facial5points, image_size=112, createEvalDB=evalDB)

            save_image(torch.from_numpy(warped_face).permute(2, 0, 1) / 255.0, os.path.join(out_folder, img_name))
            counter += 1

        print(np.mean(landmark_statistics, axis=0), np.std(landmark_statistics, axis=0))

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")


def main():
    parser = argparse.ArgumentParser(description="MTCNN alignment")
    parser.add_argument(
        "--in_folder",
        type=str,
        default="/workspace/IDiff-Face//data/ffhq_128",
        help="folder with images",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="/workspace/IDiff-Face//aligned",
        help="folder to save aligned images",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--evalDB", type=int, default=1, help="1 for eval DB alignment")

    args = parser.parse_args()
    align_images(
        args.in_folder,
        args.out_folder,
        args.batch_size,
        evalDB=args.evalDB == 1,
    )


if __name__ == "__main__":
    main()