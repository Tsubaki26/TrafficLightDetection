from torch.utils.data import Dataset
import torch
from PIL import Image

from other.utils import parse_annotation_CVAT


class TrafficLightDataset(Dataset):
    """
    アノテーションは CVAT for images 1.1 フォーマットの xml ファイルを用いる。
    """

    def __init__(self, annotation_path, img_dir_path, img_size, transforms=None):
        self.annotation_path = annotation_path
        self.img_dir_path = img_dir_path
        self.img_size = img_size  # height width
        self.transforms = transforms

        (
            self.num_images,
            self.img_path_all,
            self.label_all,
            self.bbox_all,
            self.class_to_idx,
            self.classes,
            self.colors,
        ) = parse_annotation_CVAT(self.annotation_path, img_dir_path, self.img_size)

    def __len__(self):
        """
        データセット内の合計画像枚数を取得

        Returns:
            データセット内の合計画像枚数: int
        """
        return self.num_images

    def __getitem__(self, index):
        """
        Returns:
            img: tensor
            target: "bboxes", "labels", "image_id"
        """
        img = Image.open(self.img_path_all[index])
        if self.transforms != None:
            img_tensor = self.transforms(img)

        image_id = torch.tensor([index])

        target = {}
        target["boxes"] = self.bbox_all[index]
        target["labels"] = self.label_all[index]
        target["image_id"] = image_id
        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
