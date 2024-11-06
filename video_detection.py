import numpy as np
from PIL import Image
import torch
import ffmpeg # type: ignore
import torchvision
import time

from utils import detection_one_image
from TrafficLightDataset import TrafficLightDataset

VIDEO_PATH = "./dataset2/test/drive_test_short.mp4"
TEST_ANNOTATION_PATH = "./dataset2/test/annotations.xml"
TEST_IMG_DIR_PATH = "./dataset2/test/images/"
MODEL_PATH = "./models/model_dataset2_b8_e200_lr0.005.pth"
IMG_SIZE = (1080, 1920)
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

t = time.time()
OUTPUT_VIDEO_PATH = f"out{t}.mp4"

if __name__ == "__main__":
    probe = ffmpeg.probe(VIDEO_PATH)
    video_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    ]
    width = video_streams[0]["width"]
    height = video_streams[0]["height"]
    fps = int(eval(video_streams[0]["r_frame_rate"]))

    cap = (
        ffmpeg.input(VIDEO_PATH)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True)
    )

    writer = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(IMG_WIDTH, IMG_HEIGHT),
            r=fps,
        )
        .output(OUTPUT_VIDEO_PATH, pix_fmt="yuv420p")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    test_dataset = TrafficLightDataset(
        annotation_path=TEST_ANNOTATION_PATH,
        img_dir_path=TEST_IMG_DIR_PATH,
        img_size=IMG_SIZE,
        transforms=None,
    )
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.to(DEVICE)
    model.eval()

    count = 0
    while True:
        count += 1
        in_bytes = cap.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
        out_frame = Image.fromarray(in_frame).resize(
            (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR
        )
        img_tensor = torchvision.transforms.functional.to_tensor(out_frame)

        img_tensor = img_tensor.to(DEVICE)
        input_img = [img_tensor]
        with torch.no_grad():
            prediction = model(input_img)[1]
        # print("\nBOXES: ", prediction[0]["boxes"])

        # 推論
        dimg = detection_one_image(
            image=img_tensor,
            bboxes=prediction[0]["boxes"].cpu(),
            labels=prediction[0]["labels"].cpu(),
            classes=test_dataset.classes,
            label_colors=test_dataset.colors,
            device=DEVICE,
            scores=prediction[0]["scores"].cpu(),
        )
        writer.stdin.write(dimg.astype(np.uint8).tobytes())
        # if count > 30:
        #     break
    writer.stdin.close()
    cap.wait()
    writer.wait()
