import os

import xml.etree.ElementTree as ET
import torch
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
import cv2


def parse_annotation_CVAT(annotaion_path, img_dir_path, img_size):
    """
    CVAT形式のxmlファイルを解析

    Returns:
        画像枚数: int
        画像パス配列: list
        ラベル配列: list
        Bボックス配列: tensorのリスト
    """
    print(annotaion_path)
    with open(annotaion_path, "r") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # Label 0は背景として扱われるので、背景クラスを含まない場合はオブジェクトに 0 を割り当ててはいけない。
    class_to_idx = {}
    classes = []
    colors = []
    classes.append("background")
    class_to_idx["background"] = 0
    colors.append("#ffffff")
    labels = root.findall("meta/job/labels/label")
    for i, label in enumerate(labels):
        class_name = label.find("name").text
        color = label.find("color").text
        classes.append(class_name)
        class_to_idx[class_name] = i + 1  # 背景を0にするので、＋１している
        colors.append(color)

    img_height, img_width = img_size  # height width
    image_objects = root.findall("image")

    num_images = len(image_objects)
    print("The number of images: ", num_images)
    print("----------------------------------")

    img_path_all = []
    label_all = []
    bbox_all = []
    for object in image_objects:
        image_id = object.get("id")
        image_name = object.get("name")
        original_image_width = float(object.get("width"))
        original_image_height = float(object.get("height"))

        width_ratio = img_width / original_image_width
        height_ratio = img_height / original_image_height

        labels = []
        bboxes = []
        for box in object.findall("box"):
            label = box.get("label")
            xmin = float(box.get("xtl")) * width_ratio
            ymin = float(box.get("ytl")) * height_ratio
            xmax = float(box.get("xbr")) * width_ratio
            ymax = float(box.get("ybr")) * height_ratio

            labels.append(class_to_idx[label])
            bboxes.append([xmin, ymin, xmax, ymax])

        labels = torch.tensor(labels)
        bboxes = torch.tensor(bboxes)

        image_path = os.path.join(img_dir_path, image_name)
        img_path_all.append(image_path)
        label_all.append(labels)
        bbox_all.append(bboxes)

        print("image_id: ", image_id)
        print("image_name: ", image_name)
        print("original_image_width: ", original_image_width)
        print("original_image_height: ", original_image_height)
        print("labels: ", labels)
        print("bboxes: ", bboxes)
        print("----------------------------------")

    print("classes: ", classes)
    print("class_to_idx: ", class_to_idx)
    print("colors: ", colors)
    print("img_path_all: ", img_path_all)
    print("label_all: ", label_all)
    print("bbox_all: ", bbox_all)

    return num_images, img_path_all, label_all, bbox_all, class_to_idx, classes, colors


def tensor_img2numpy_img(image, device):
    """
    Tensor 型の image を numpy array に変換
    引数:
        ・Tensor型画像
    Returns:
        ・Numpy型画像
    """
    image_numpy = image.detach().cpu().permute(1, 2, 0).numpy().copy()
    image = image.to(device)
    return image_numpy


def display_img_bbox(image, bboxes, labels, classes, label_colors, device, scores=None):
    """
    バウンディングボックス付きの画像を表示
    引数:
        Tensor型画像
        画像内のバウンディングボックスリスト
        画像内のラベルリスト
        クラスリスト
        クラス色リスト
        デバイス
    """
    threshold = 0.8
    image_numpy = tensor_img2numpy_img(image, device)

    plt.imshow(image_numpy)

    labels = list(labels)
    for i, box in enumerate(bboxes):
        hex_color = label_colors[labels[i]]
        if scores == None:
            box = list(box)
            rect = patches.Rectangle(
                xy=(box[0], box[1]),
                width=box[2] - box[0],
                height=box[3] - box[1],
                ec=hex_color,
                fill=False,
            )
            plt.gca().add_patch(rect)
            plt.text(box[0], box[1], classes[labels[i]])
        elif scores[i] > threshold:
            # bounding box作成
            box = list(box)
            rect = patches.Rectangle(
                xy=(box[0], box[1]),
                width=box[2] - box[0],
                height=box[3] - box[1],
                ec=hex_color,
                fill=False,
            )
            plt.gca().add_patch(rect)
            plt.text(box[0], box[1], f"{classes[labels[i]]} ({scores[i]:.2f})")
    plt.title("Groundtruth bounding boxes")
    plt.show()


def hex_to_rgb_tuple(hex_str):
    """
    16進数カラーコードをRGBのタプルに変換する関数
    Returns:
      RGBのタプル (R, G, B)
    """

    hex_str = hex_str.strip("#")
    if len(hex_str) != 6:
        raise ValueError("Invalid hex string")
    rgb_tuple = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    return rgb_tuple


def attach_traffic_light(image_numpy, label, is_right_forward_left, color):
    # 1920x1080の画像用に設定
    # background
    cv2.rectangle(image_numpy, (1540, 880), (1900, 980), (50, 50, 50), thickness=-1)
    cv2.rectangle(image_numpy, (1540, 983), (1900, 1070), (50, 50, 50), thickness=-1)
    colors = ["#666666", "#666666", "#666666", "#666666", "#666666", "#666666"]
    if label is not None:
        colors[label - 1] = color
    if is_right_forward_left[0] == True:
        colors[3] = "#34d1b7"
    if is_right_forward_left[1] == True:
        colors[4] = "#34d1b7"
    if is_right_forward_left[2] == True:
        colors[5] = "#34d1b7"

    for i, c in enumerate(colors):
        color_tuple = hex_to_rgb_tuple(c)
        if i == 0:  # Green
            cv2.circle(image_numpy, (1600, 930), 40, color_tuple, thickness=-1)
        if i == 1:  # Yellow
            cv2.circle(image_numpy, (1720, 930), 40, color_tuple, thickness=-1)
        if i == 2:  # Red
            cv2.circle(image_numpy, (1840, 930), 40, color_tuple, thickness=-1)
        if i == 3:  # Right
            cv2.arrowedLine(
                image_numpy,
                (1800, 1026),
                (1880, 1026),
                color_tuple,
                thickness=10,
                tipLength=0.5,
            )
        if i == 4:  # Forward
            cv2.arrowedLine(
                image_numpy,
                (1720, 1060),
                (1720, 993),
                color_tuple,
                thickness=10,
                tipLength=0.5,
            )
        if i == 5:  # Left
            cv2.arrowedLine(
                image_numpy,
                (1640, 1026),
                (1560, 1026),
                color_tuple,
                thickness=10,
                tipLength=0.5,
            )

    return image_numpy


def detection_one_image(
    image, bboxes, labels, classes, label_colors, device, scores=None
):
    """
    バウンディングボックス付きの画像を表示
    引数:
        Tensor型画像
        画像内のバウンディングボックスリスト
        画像内のラベルリスト
        クラスリスト
        クラス色リスト
        デバイス
    """
    threshold = 0.7
    image_numpy = tensor_img2numpy_img(image, device)
    image_numpy = image_numpy * 255

    labels = list(labels)
    box_areas = []
    is_right_forward_left = [False, False, False]

    for i, box in enumerate(bboxes):
        hex_color = label_colors[labels[i]]
        color = hex_to_rgb_tuple(hex_color)
        if scores[i] > threshold:
            box_area = (int(box[2].item()) - int(box[0].item())) * (
                int(box[3].item()) - int(box[1].item())
            )
            box_areas.append(box_area)
            if labels[i] > 3:  # right, forward, left
                is_right_forward_left[labels[i] - 4] = True
            # bounding box作成
            box = list(box)
            cv2.rectangle(
                image_numpy,
                (int(box[0].item()), int(box[1].item())),
                (int(box[2].item()), int(box[3].item())),
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image_numpy,
                f"{classes[labels[i]]} ({scores[i]:.2f})",
                (int(box[0].item()), int(box[1].item())),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )
    if len(box_areas) > 0:
        max_box_area_index = box_areas.index(max(box_areas))
        print(max_box_area_index)
        max_box_color = label_colors[labels[max_box_area_index]]
        max_box_color = hex_to_rgb_tuple(max_box_color)
        label = labels[max_box_area_index]
        color = label_colors[labels[max_box_area_index]]
    else:
        label = None
        color = None

    image_numpy = attach_traffic_light(
        image_numpy,
        label,
        is_right_forward_left,
        color,
    )
    return image_numpy


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    ファインチューニングを行う場合、学習率が大きい状態で学習を始めてしまうと
    学習済みのパラメータが壊れてしまう。
    warmup は最初に低い学習率を設定し、徐々に増加させていき、その後減少させる。
    """

    def warmup_lr_lambda(current_iter):
        # lr_lambda: lr_scheduler.LambdaLR では lr_lambda の関数に基づいて学習率が更新される。
        if current_iter >= warmup_iters:
            return 1.0

        alpha = float(current_iter) / warmup_iters

        # warm_factor > 1: 徐々に傾きが減少し、warmup_iters 到達時に１に収束
        # warm_factor < 1: 徐々に傾きが増加し、warmup_iters 到達時に１に収束
        return warmup_factor * (1 - alpha) + alpha

    return optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warmup_lr_lambda)


def train_one_epoch(model, optimizer, dataloader, device, epoch):
    """
    1エポックの学習

    Returns:
        各バッチの train loss の平均値
    """
    train_loss = 0
    display_loss_interval = 50

    model.train()  # 学習モード

    lr_shceduler = None

    # 最初だけ warmup
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_shceduler = warmup_lr_scheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            warmup_factor=warmup_factor,
        )

    # Dataloader のループ
    for idx, (images, targets) in enumerate(dataloader):
        # image と targets を device に移動
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # for image in images:
        #     img_pil = to_pil_image(image)
        # plt.figure()
        # plt.imshow(img_pil)
        # plt.show()

        loss_dict = model(images, targets)
        # print("loss_dict: ", loss_dict)
        # if torch.isnan(loss_dict).any():
        #     print(f"NaN detected in loss at epoch {epoch}")

        # loss の合計
        loss_sum = sum(loss for loss in loss_dict.values())
        train_loss += loss_sum.item()

        # 50バッチごとに loss の平均を表示
        if (idx + 1) % display_loss_interval == 0:
            average_loss = train_loss / (idx + 1)
            print(f"Batch: {idx+1}/{len(dataloader)} | Loss: {average_loss:.5f}")

        optimizer.zero_grad()
        loss_sum.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if lr_shceduler != None:
            lr_shceduler.step()

        # GPUキャッシュをクリア
        torch.cuda.empty_cache()

    # 1エポック終了時の loss の平均を表示
    train_loss = train_loss / len(dataloader)
    print(f"Batch: {idx+1}/{len(dataloader)} | Loss: {train_loss:.5f}")

    return train_loss


def evaluate(model, dataloader, device):
    """
    推論

    Returns:
        各バッチの eval_loss の平均値
    """
    eval_loss = 0
    display_loss_interval = 50

    model.to(device)

    for idx, (images, targets) in enumerate(dataloader):
        # image と targets を device に移動
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 推論モード
        # model.eval()よりもメモリ効率が改善された推論モードらしい
        with torch.inference_mode():
            loss_dict = model(images, targets)

        loss_sum = sum(loss for loss in loss_dict.values())
        eval_loss += loss_sum.item()

        if (idx + 1) % display_loss_interval == 0:
            average_loss = eval_loss / (idx + 1)
            print(f"Batch: {idx+1}/{len(dataloader)} | Loss: {average_loss:.5f}")
            print(f"Loss Dict: {loss_dict}")

        # GPUキャッシュをクリア
        torch.cuda.empty_cache()

    # 1エポック終了時の loss の平均を表示
    eval_loss = eval_loss / len(dataloader)
    print(f"Batch: {idx+1}/{len(dataloader)} | Loss: {eval_loss:.5f}")

    return eval_loss
