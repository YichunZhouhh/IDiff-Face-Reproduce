import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        """
        self.data = [
            {'source': 'source_001.png', 'target': 'target_001.png', 'prompt': 'a red apple'},
            {'source': 'source_002.png', 'target': 'target_002.png', 'prompt': 'a blue car'},
            {'source': 'source_003.png', 'target': 'target_003.png', 'prompt': 'a white cat'},
         ... 50000 条记录
            ]
        """
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Source (hint): 条件图像（控制信号）
        Target (jpg): 目标图像（生成目标）
        Prompt (txt): 文本提示（可选的额外条件）
        """
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename) # 返回numpy数组BGR色彩空间（H,W,3）
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

