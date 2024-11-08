import torch
from torchvision import transforms

from train.TrafficLightDataset import TrafficLightDataset
from other.utils import display_img_bbox

# TEST_ANNOTATION_PATH = "./mini_train_dataset/annotations.xml"
# TEST_IMG_DIR_PATH = "./mini_train_dataset/images/"
# TEST_ANNOTATION_PATH = "./dataset/train/annotations.xml"
# TEST_IMG_DIR_PATH = "./dataset/train/images/"
TEST_ANNOTATION_PATH = "./dataset2/test/annotations.xml"
TEST_IMG_DIR_PATH = "./dataset2/test/images/"
MODEL_PATH = "./models/model_dataset2_b8_e200_lr0.005.pth"
IMG_SIZE = (1080, 1920)
DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")


model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.to(DEVICE)

resize = transforms.Resize(IMG_SIZE)
to_tensor = transforms.ToTensor()
test_transform = transforms.Compose([resize, to_tensor])
test_dataset = TrafficLightDataset(
    annotation_path=TEST_ANNOTATION_PATH,
    img_dir_path=TEST_IMG_DIR_PATH,
    img_size=IMG_SIZE,
    transforms=test_transform,
)


model.eval()
imgs = []
for i in range(test_dataset.__len__()):
    img, target = test_dataset.__getitem__(i)
    imgs.append(img.to(DEVICE))
    if i > 10:
        break

with torch.no_grad():
    predictions = model(imgs)[1]
print(predictions)

# GPUキャッシュをクリア
torch.cuda.empty_cache()

for i, prediction in enumerate(predictions):
    display_img_bbox(
        image=imgs[i],
        bboxes=prediction["boxes"].cpu(),
        labels=prediction["labels"].cpu(),
        classes=test_dataset.classes,
        label_colors=test_dataset.colors,
        device=DEVICE,
        scores=prediction["scores"].cpu(),
    )
