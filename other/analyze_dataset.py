import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


TRAIN_ANNOTATION_PATH = "./dataset2/train/annotations.xml"
TEST_ANNOTATION_PATH = "./dataset2/test/annotations.xml"


def analyze_dataset(annotaion_path):
    """
    CVAT形式のxmlファイルを解析
    """
    with open(annotaion_path, "r") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    class_num = {}
    class_to_color = {}
    classes = []
    colors = []
    num_objects = 0
    left = []
    height = []
    dataset_type = root.findtext("meta/job/subset")
    print(dataset_type)
    labels = root.findall("meta/job/labels/label")
    for i, label in enumerate(labels):
        left.append(i)
        class_name = label.find("name").text
        color = label.find("color").text
        # classes.append(class_name)
        class_num[class_name] = 0
        class_to_color[class_name] = color
        # colors.append(color)

    image_objects = root.findall("image")

    num_images = len(image_objects)

    for object in image_objects:
        for box in object.findall("box"):
            num_objects += 1
            label = box.get("label")
            class_num[label] += 1

    class_num_sorted = sorted(class_num.items(), key=lambda x: x[1], reverse=True)
    for key, value in class_num_sorted:
        classes.append(key)
        height.append(value)
        colors.append(class_to_color[key])

    print("==================================")
    print("Dataset name: ", annotaion_path)
    print("The number of images: ", num_images)
    print("The number of objects: ", num_objects)
    print("classes: ", classes)
    print("colors: ", colors)
    print("class num: ", class_num)
    print("class num sorted: ", class_num_sorted)
    print("\n")

    # データの可視化
    plt.subplot(121)
    plt.bar(left, height, tick_label=classes, align="center", color=colors)
    plt.subplot(122)
    plt.pie(
        height,
        counterclock=False,
        startangle=90,
        colors=colors,
        labels=classes,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        autopct="%1.1f%%",
        pctdistance=0.8,
    )
    plt.suptitle(dataset_type)
    plt.show()


if __name__ == "__main__":
    analyze_dataset(TRAIN_ANNOTATION_PATH)
    analyze_dataset(TEST_ANNOTATION_PATH)
