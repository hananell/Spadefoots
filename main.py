import os.path
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

from movements import detect_movements
from spadefoots import detect_spadefoots

month_dict = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12,
    'ינואר': 1,
    'פברואר': 2,
    'מרץ': 3,
    'אפריל': 4,
    'מאי': 5,
    'יוני': 6,
    'יולי': 7,
    'אוגוסט': 8,
    'ספטמבר': 9,
    'אוקטובר': 10,
    'נובמבר': 11,
    'דצמבר': 12,
    'îàé': 5,
    'éåìé': 6,
    'éåðé': 7,
    'àåâåñè': 8,
    'àôøéì': 9,
}


# calculate iou between two given boxes
def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# draw spadefoots in red, previous regions of spadefoots in blue, movements in green. save image to with_movements or without_movements
def draw_save(imgfile, spadefoots_prev, spadefoots_cur, movements):
    # init image
    img = Image.open(imgfile)
    figi = plt.figure()
    fig, ax = plt.subplots()
    ax.imshow(img)

    # plot regions of previous spadefoots in blue
    for bbox in spadefoots_prev:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='b', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    # plot spadefoots in red
    for bbox in spadefoots_cur:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    # plot movements in green
    for bbox in movements:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='g', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    # default destination is without_movements. change if there is an overlapping of a spadefoot and a movement
    dest = 'without_movements'
    for spadefoot in spadefoots_prev + spadefoots_cur:
        for movement in movements:
            if iou(spadefoot, movement) > 0.05:
                dest = 'with_movements'

    plt.savefig(f"{dest}/{imgfile[6:].replace('/', '_')}")
    plt.close()
    plt.close(figi)


# return timestamp of a given image, for sorting
def timestamp_image(imgFile):
    try:
        tokens = os.path.basename(imgFile).split('.')[0].split(' ')
        day = int(tokens[0][8:])
        year, month = (int(tokens[1]), month_dict[tokens[2]]) if tokens[2] in month_dict else (int(tokens[2]), month_dict[tokens[1]])
        hours, minutes, seconds = list(map(int, tokens[4].split('`')))
        return (year, month, day, hours, minutes, seconds)
    except Exception:
        if 'Banna' in imgFile and '(' in imgFile and ')' in imgFile:
            num = int(imgFile[imgFile.index('(') + 1:imgFile.index(')')])
            return (0, 0, 0, 0, 0, num)

        if 'Janice' in imgFile:
            num = int(imgFile[imgFile.index('t') + 1:-4]) if len(imgFile[imgFile.index('t') + 1:-4]) <= 3 else 11
            return (0, 0, 0, 0, 0, num)

        if 'Thumper' in imgFile:
            num = int(imgFile[imgFile.index('ot') + 2:-4])
            return (0, 0, 0, 0, 0, num)

        if '(1)' in imgFile:
            tokens = os.path.basename(imgFile).split('(1)')[0].split(' ')
            day = int(tokens[0][8:])
            year, month = (int(tokens[1]), month_dict[tokens[2]]) if tokens[2] in month_dict else (int(tokens[2]), month_dict[tokens[1]])
            hours, minutes, seconds = list(map(int, tokens[4].split('`')))
            return (year, month, day, hours, minutes, seconds)

        print(f"timestamp problem:  {camera}/{imgFile}")
        return (0, 0, 0, 0, 0, os.path.getmtime(imgFile))


if __name__ == "__main__":
    for camera in glob('data/comp*/camera*'):
        # if 'Shenzi' not in camera:
        #     continue

        # make sorted list of images, and make them full address
        images = sorted(os.listdir(camera), key=lambda img: timestamp_image(f"{camera}/{img}"))
        images = [f"{camera}/{img}" for img in images]

        # treat first two images separately, to draw_save the first
        img1, img2 = images[0], images[1]
        movements = detect_movements(img1, img2)
        spadefoots1, spadefoots2 = detect_spadefoots(img1), detect_spadefoots(img2)
        draw_save(img1, [], spadefoots1, movements)
        draw_save(img2, spadefoots1, spadefoots2, movements)

        # for each two images: detect movements, detect spadefoots, draw_save the second. if there is any overlapping it will get saved into with_movements
        for i in tqdm(range(1, len(images) - 1), postfix=camera):
            img1, img2 = images[i], images[i + 1]
            movements = detect_movements(img1, img2)
            spadefoots1 = spadefoots2
            spadefoots2 = detect_spadefoots(img2)
            draw_save(img2, spadefoots1, spadefoots2, movements)
