import os
import time
import random

import matplotlib.pyplot as plt
import torch
from torch import optim
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader

from TrafficLightDataset import TrafficLightDataset, collate_fn
from model import fasterrcnn_backbone_resnet101
from utils import display_img_bbox
from utils import train_one_epoch, evaluate
from early_stopping import EarlyStopping

DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
print(DEVICE)

# TRAIN_ANNOTATION_PATH = "./mini_train_dataset/annotations.xml"
# TRAIN_IMG_DIR_PATH = "./mini_train_dataset/images/"
# TEST_ANNOTATION_PATH = "./mini_train_dataset/annotations.xml"
# TEST_IMG_DIR_PATH = "./mini_train_dataset/images/"
# TRAIN_ANNOTATION_PATH = "./dataset/train/annotations.xml"
# TRAIN_IMG_DIR_PATH = "./dataset/train/images/"
# TEST_ANNOTATION_PATH = "./dataset/test/annotations.xml"
# TEST_IMG_DIR_PATH = "./dataset/test/images/"
TRAIN_ANNOTATION_PATH = "./dataset2/train/annotations.xml"
TRAIN_IMG_DIR_PATH = "./dataset2/train/images/"
TEST_ANNOTATION_PATH = "./dataset2/test/annotations.xml"
TEST_IMG_DIR_PATH = "./dataset2/test/images/"


DATASET_NAME = "dataset2"

# ハイパーパラメータ
IMG_SIZE = (1080, 1920)  # Image Size: (height, width)
NUM_EPOCHS = 200
BATCH_SIZE = 2
# NUM_WORKERS = os.cpu_count()  # コア数を設定。マルチプロセスで高速化
LR = 0.005
LR_MOMENTUM = 0.9
LR_DECAY_RATE = 0.0005
LR_SCHED_STEP_SIZE = 0.1
LR_SCHED_GAMMA = 0.1

MODEL_PATH = f"./models/model_{DATASET_NAME}_b{BATCH_SIZE}_e{NUM_EPOCHS}_lr{LR}.pth"
LEARNING_CURVE_PATH = (
    f"./models/learning_plot_{DATASET_NAME}_b{BATCH_SIZE}_e{NUM_EPOCHS}_lr{LR}.png"
)


if __name__ == "__main__":

    # GPUキャッシュをクリア
    torch.cuda.empty_cache()

    # Transforms
    # transformにランダム要素を入れると、データをロードするたびに
    # 異なる画像処理が施され、データ拡張になる。
    resize = transforms.Resize(IMG_SIZE)
    to_tensor = transforms.ToTensor()
    contrast = transforms.RandomAutocontrast(p=0.5)
    gaussian_blur = transforms.GaussianBlur(
        kernel_size=random.randrange(7, 12, 4), sigma=(0.1, 2.0)
    )
    train_transform = transforms.Compose(
        [
            contrast,
            gaussian_blur,
            resize,
            to_tensor,
        ]
    )
    test_transform = transforms.Compose([resize, to_tensor])

    # Train Dataset
    train_dataset = TrafficLightDataset(
        annotation_path=TRAIN_ANNOTATION_PATH,
        img_dir_path=TRAIN_IMG_DIR_PATH,
        img_size=IMG_SIZE,
        transforms=train_transform,
    )

    # Test Dataset
    test_dataset = TrafficLightDataset(
        annotation_path=TEST_ANNOTATION_PATH,
        img_dir_path=TEST_IMG_DIR_PATH,
        img_size=IMG_SIZE,
        transforms=test_transform,
    )

    # Train Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,  # 高速化
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,  # 高速化
    )
    # Test Dataloader
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        # num_workers=NUM_WORKERS,  # 高速化
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,  # 高速化
    )

    # img, target = train_dataset.__getitem__(9)
    # display_img_bbox(
    #     image=img,
    #     bboxes=target["boxes"],
    #     labels=target["labels"],
    #     classes=train_dataset.classes,
    #     label_colors=train_dataset.colors,
    #     device=DEVICE,
    # )

    # Model
    NUM_CLASSES = len(train_dataset.class_to_idx)
    model = fasterrcnn_backbone_resnet101(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_layer_on_fpn=False,
    )
    model.to(DEVICE)

    # Optimizer
    params = [
        p for p in model.parameters() if p.requires_grad
    ]  # 学習するパラメータを取得
    optimizer = optim.SGD(
        params=params,
        lr=LR,
        momentum=LR_MOMENTUM,
        weight_decay=LR_DECAY_RATE,
    )
    # ReduceLROnPlateau: 指定した評価指標の改善が停滞した場合に、学習率を低減させる。
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.1,
        patience=2,  # データ数が少ない場合、この値を大きくすると loss が発散した
        verbose=False,
    )

    # Training Loop
    results = {
        "train_loss": [],
        "test_loss": [],
    }

    torch.autograd.set_detect_anomaly(True)
    earlystopping = EarlyStopping(patience=2, verbose=False, path=MODEL_PATH)

    for epoch in range(NUM_EPOCHS):
        current_epoch = epoch + 1
        print(f"\nEpoch: {current_epoch}/{NUM_EPOCHS}")

        start_time = time.time()

        # Training
        print("\nTraining...")
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            device=DEVICE,
            epoch=epoch,
        )

        lr_scheduler.step(train_loss)

        # Testing
        print("\nTesting...")
        test_loss = evaluate(
            model=model,
            dataloader=test_dataloader,
            device=DEVICE,
        )

        # 1エポック終了時の loss を表示
        print(
            f"\nEpoch: {current_epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.5f} | Test Loss: {test_loss:.5f}"
        )

        end_time = time.time()
        epoch_time = end_time - start_time
        hours, remainder = divmod(epoch_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        # 1エポックにかかった時間の表示
        print(
            f"Time taken for epoch {epoch}: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
        )

        # Early stopping
        earlystopping(test_loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

        # results の更新
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # GPUキャッシュをクリア
    torch.cuda.empty_cache()

    # Torch Script形式で保存
    # model_scripted = torch.jit.script(model)
    # model_scripted.save(
    #     f"./models/model_script_{DATASET_NAME}_b{BATCH_SIZE}_e{NUM_EPOCHS}_lr{LR}.pth"
    # )
    # print(
    #     f"saved model as ./models/model_script_{DATASET_NAME}_b{BATCH_SIZE}_e{NUM_EPOCHS}_lr{LR}.pth"
    # )

    # img0, target = train_dataset.__getitem__(0)
    # img1, target = train_dataset.__getitem__(1)
    # imgs = [img0.to(DEVICE), img1.to(DEVICE)]
    # model.eval()
    # with torch.no_grad():
    #     predictions = model(imgs)
    # print(predictions)
    # for i, prediction in enumerate(predictions):
    #     display_img_bbox(
    #         image=imgs[i],
    #         bboxes=prediction["boxes"],
    #         labels=prediction["labels"],
    #         classes=train_dataset.classes,
    #         label_colors=train_dataset.colors,
    #         device=DEVICE,
    #     )

    # 学習過程の表示
    plt.plot(range(len(results["train_loss"])), results["train_loss"], label="Train")
    plt.plot(range(len(results["test_loss"])), results["test_loss"], label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.savefig(LEARNING_CURVE_PATH)
    print(f"saved learning plot figure to {LEARNING_CURVE_PATH}")
    plt.show()
    plt.close()
