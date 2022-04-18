import numpy as np
import torch
import time
from torchvision import transforms
from PIL import Image
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best_spadefoots5.pt").to(device).eval()


# from yolov5 repository
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# from yolov5 repository
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


# # from yolov5 repository
def non_max_suppression(prediction, conf_thres, iou_thres, classes, agnostic=True):
    """ Performs Non-Maximum Suppression (NMS) on inference results
        prediction.shape = (1, number_of_anchors, 85)
        prediction.shape[2] = 85 = len(box_coordinates)+len(probability)+len(class_conf) = 4+1+80
    Returns:
        detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = None
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = True  # use merge-NMS

    assert isinstance(conf_thres, list), f"Invalid type: type(conf_thres)={type(conf_thres)}"
    assert len(conf_thres) == len(iou_thres) == len(classes), \
        f"Invalid arguments. len(conf_thres)={len(conf_thres)}, len(iou_thres)={len(iou_thres)} \
    and len(classses)={len(classes)} should be the same"

    output = list()
    t = time.time()
    for cls_idx, conf_t, iou_t in zip(classes, conf_thres, iou_thres):

        class_filter = torch.argmax(prediction[..., 5:85], dim=2) == cls_idx
        conf_filter = prediction[..., 4] > conf_t
        xc = class_filter & conf_filter

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])

            i, j = (x[:, 5:] > conf_t).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_t)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_t  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output.append(x[i])
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

    return output


# detect spadefoots in given frame
def detect_spadefoots(frame):
    frame = transforms.ToTensor()(np.asarray(Image.open(frame)))
    frame = torch.unsqueeze(frame, 0)
    detectionsLists = model(frame, augment=True)
    detectionsLists = non_max_suppression(detectionsLists, conf_thres=[0.8], iou_thres=[0.2], classes=[0], agnostic=True)
    detectionsList = detectionsLists[0] if detectionsLists else []
    detections = [[val.item() for val in detectionsList[i]][:4] for i in range(len(detectionsList))]
    return detections
