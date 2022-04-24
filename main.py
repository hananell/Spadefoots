import os.path
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm
import xlwt
from xlwt import Workbook
import shutil

from movements import detect_movements
from spadefoots import detect_spadefoots

data_dir = 'data'
results_dir = 'results'
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
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# draw spadefoots in red, previous regions of spadefoots in blue, movements in green. save image to with_movements or without_movements.
# return true if a spadefoot has moved, else false
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

    # save image to its place in results
    plt.savefig(f"{results_dir}/{camera[camera.index('/') + 1:].replace('/', '_')}/{dest}/{imgfile[imgfile.index('Snap'):]}")
    plt.close()
    plt.close(figi)

    return True if dest == 'with_movements' else False


def just_save(imgfile, spadefoots_prev, spadefoots_cur, movements):
    # default destination is without_movements. change if there is an overlapping of a spadefoot and a movement
    dest = 'without_movements'
    for spadefoot in spadefoots_prev + spadefoots_cur:
        for movement in movements:
            if iou(spadefoot, movement) > 0.05:
                dest = 'with_movements'

    # save image to its place in results
    n = f"{results_dir}/{camera[camera.index('/') + 1:].replace('/', '_')}/{dest}/{imgfile[imgfile.index('Snap'):]}"
    shutil.copy2(imgfile, f"{results_dir}/{camera[camera.index('/') + 1:].replace('/', '_')}/{dest}/{imgfile[imgfile.index('Snap'):]}")

    return True if dest == 'with_movements' else False


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


# init dirs and xls file for saving results
def init_results():
    # make dirs
    if os.path.isdir(f'{results_dir}'):
        shutil.rmtree(f'{results_dir}')
    os.mkdir('results')
    for camera in glob(f'{data_dir}/comp*/camera*'):
        os.mkdir(f"{results_dir}/{camera[camera.index('/') + 1:].replace('/', '_')}")
        os.mkdir(f"{results_dir}/{camera[camera.index('/') + 1:].replace('/', '_')}/with_movements")
        os.mkdir(f"{results_dir}/{camera[camera.index('/') + 1:].replace('/', '_')}/without_movements")

    # make xls file
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    bold = xlwt.easyxf('font: bold 1')
    for i, camera in enumerate(glob(f'{data_dir}/comp*/camera*')):
        sheet1.write(0, 2 * i, f"{camera[camera.index('/') + 1:]} with_movements", bold)
        sheet1.write(0, 2 * i + 1, f"{camera[camera.index('/') + 1:]} without_movements", bold)
    return wb, sheet1


if __name__ == "__main__":
    wb, sheet = init_results()

    for camera_ind, camera in enumerate(glob(f'{data_dir}/comp*/camera*')):
        with_movements_count, without_movements_count = 0, 0  # line to write in xls file

        # make sorted list of images, and make them full address
        images = sorted(os.listdir(camera), key=lambda img: timestamp_image(f"{camera}/{img}"))
        images = [f"{camera}/{img}" for img in images]

        # treat first image separately, comparing it to next image for movements instead of previous one
        img1, img2 = images[0], images[1]
        movements = detect_movements(img1, img2)
        spadefoots2 = detect_spadefoots(img1)  # name is spadefoots2 so in the next iteration it will be taken as previous detections
        img1_result = draw_save(img1, [], spadefoots2, movements)
        column = camera_ind * 2 if img1_result else camera_ind * 2 + 1
        if img1_result:
            with_movements_count += 1
        else:
            without_movements_count += 1
        line = with_movements_count if img1_result else without_movements_count

        sheet.write(line, column, os.path.basename(img1))

        # for each two images: detect movements, detect spadefoots, draw_save the second. if there is any overlapping it will get saved into with_movements
        for img_ind in tqdm(range(len(images) - 1), postfix=camera[camera.index('/') + 1:]):
            img1, img2 = images[img_ind], images[img_ind + 1]
            movements = detect_movements(img1, img2)
            spadefoots1 = spadefoots2
            spadefoots2 = detect_spadefoots(img2)
            img2_result = just_save(img2, spadefoots1, spadefoots2, movements)
            column = camera_ind * 2 if img2_result else camera_ind * 2 + 1
            if img2_result:
                with_movements_count += 1
            else:
                without_movements_count += 1
            line = with_movements_count if img2_result else without_movements_count
            sheet.write(line, column, os.path.basename(img2))  # +2 because 0 is title and 1 is taken by first image

    wb.save(f'{results_dir}/results.xls')