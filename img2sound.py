import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile


SAMPLE_RATE = 22050

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert images into sounds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the image.",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Show the images processed.",
    )

    return parser.parse_args()

def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img


def show_img(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bgr_to_hsv(bgr_img):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return hsv_img


def rgb_to_bgr(rgb_img):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr_img


def show_imgs(img_dict):
    # Plot the image
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    names = ["BGR", "RGB", "HSV"]
    imgs = [img_dict["bgr"], img_dict["rgb"], img_dict["hsv"]]
    i = 0
    for elem in imgs:
        axs[i].title.set_text(names[i])
        axs[i].imshow(elem)
        axs[i].grid(False)
        i += 1
    plt.show()


def extract_hues(hsv_img):
    hues = []
    for i in range(hsv_img.shape[0]):
        for j in range(hsv_img.shape[1]):
            hues.append(hsv_img[i][j][0])
    return hues


def hues_to_freq(hues):
    scale_freqs = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 415.30]
    thresholds = [26, 52, 78, 104, 128, 154, 180]
    notes = []
    for h in hues:
        if h <= thresholds[0]:
            notes.append(scale_freqs[0])
        elif h > thresholds[0] and h <= thresholds[1]:
            notes.append(scale_freqs[1])
        elif h > thresholds[1] and h <= thresholds[2]:
            notes.append(scale_freqs[2])
        elif h > thresholds[2] and h <= thresholds[3]:
            notes.append(scale_freqs[3])
        elif h > thresholds[3] and h <= thresholds[4]:
            notes.append(scale_freqs[4])
        elif h > thresholds[4] and h <= thresholds[5]:
            notes.append(scale_freqs[5])
        elif h > thresholds[5] and h <= thresholds[6]:
            notes.append(scale_freqs[6])
        else:
            notes.append(scale_freqs[0])
    return notes


def notes_to_audio(notes):
    song = np.array([])
    sample_rate = 22050
    sample_time = 0.1
    t = np.linspace(0, sample_time, int(sample_time * SAMPLE_RATE), endpoint=False)
    for i in notes:
        note = np.sin(np.pi * i * t * 2) * 0.5
        song = np.concatenate([song, note])
    return song


def check_input(path):
    if not os.path.exists(img_path):
        print("{} does not exist!".format(img_path))
        exit(1)
    if not img_path.endswith(".jpg"):
        print("{} is not a JPG file!".format(img_path))
        exit(2)
    return True

def main():
    args = parse_args()

    print("Load image")
    img_path = args.input
    check_input(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Load the image as RGB
    rgb_img = load_image(img_path)
    # Convert the image from RGB to BGR and HSV color systems
    bgr_img = rgb_to_bgr(rgb_img)
    hsv_img = bgr_to_hsv(bgr_img)
    if args.show:
        show_imgs({"rgb": rgb_img, "bgr": bgr_img, "hsv": hsv_img})
    hues = extract_hues(hsv_img)
    
    notes = hues_to_freq(hues[1228300:1229300])
    song = notes_to_audio(notes)

    wavfile.write(img_name + ".wav", rate=SAMPLE_RATE, data=song.T.astype(np.float32))


if __name__ == "__main__":
    main()
