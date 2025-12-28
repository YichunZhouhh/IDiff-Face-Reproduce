# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage import transform as trans

# 定义112×112标准人脸图像中5个关键点的理想位置（训练用）这些是ArcFace模型训练时使用的标准面部特征点位置。所有人脸都需要对齐到这些位置，确保模型输入的一致性
ARCFACE_REFERENCE_POINTS = np.array(
    [
        [30.2946, 51.6963], # 左眼中心
        [65.5318, 51.5014], # 右眼中心
        [48.0252, 71.7366], # 笔尖
        [33.5493, 92.3655], # 左嘴角
        [62.7299, 92.2041], # 右嘴角
    ],
    dtype=np.float32,
)

# 定义评估数据集使用的参考点（整体向右偏移约8像素）评估时可能需要不同的裁剪策略，留出更多边缘信息或模拟不同的对齐条件
ARCFACE_EVAL_REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# 估计将检测到的人脸关键点对齐到标准位置所需的变换矩阵
# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, createEvalDB=False):
    """estimate the transformation matrix
    :param lmk: detected landmarks
    :param image_size: resulting image size (default=112)
    :param createEvalDB: (boolean) crop an evaluation or training dataset
    :return: transformation matrix M and index
    """
    assert lmk.shape == (5, 2)
    assert image_size == 112
    
    tform = trans.SimilarityTransform() # 相似变换保持形状，只改变位置、旋转、缩放，适合人脸对齐（不会扭曲面部）
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1) # 添加全为1的第三列 [5,3]
    min_M = []
    min_index = []
    min_error = float("inf")
    if createEvalDB:
        src = ARCFACE_EVAL_REFERENCE_POINTS
    else:
        src = ARCFACE_REFERENCE_POINTS
    src = np.expand_dims(src, axis=0) # (1,5,2)添加一个第0维，可扩展设计，防止将来尝试多个参考点集

    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i]) # 估计从检测关键点lmk到参考点src[i]的相似变换
        M = tform.params[0:2, :] # 提取2*3的仿射矩阵，tform.params是3*3矩阵，只需前两行用于opencv的warpAffine函数
        results = np.dot(M, lmk_tran.T) # [2,3]*[3,5],验证变换效果
        results = results.T # [5,2]，每一行是一个点的(x,y)
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1))) 
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


# norm_crop from Arcface repository (insightface/recognition/common/face_align.py)
# 将原始图像根据检测到的关键点对齐裁剪到标准尺寸
def norm_crop(img, landmark, image_size=112, createEvalDB=False):
    """transforms image to match the landmarks with reference landmarks
    :param landmark: detected landmarks
    :param image_size: resulting image size (default=112)
    :param createEvalDB: (boolean) crop an evaluation or training dataset
    :return: transformed image
    """
    M, pose_index = estimate_norm(
        landmark, image_size=image_size, createEvalDB=createEvalDB
    )
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0) # 最后一个参数意思是边界填充黑色
    return warped # 返回对齐后的112*112图像