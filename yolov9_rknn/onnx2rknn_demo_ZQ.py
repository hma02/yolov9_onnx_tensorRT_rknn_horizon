import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
from math import exp

ONNX_MODEL = './best.onnx'
RKNN_MODEL = './best.rknn'
DATASET = './dataset.txt'

QUANTIZE_ON = True

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush', 'face']

class_num = len(CLASSES)

meshgrid = []

input_imgH = 320
input_imgW = 320

headNum = 3
strides = [8, 16, 32]
mapSize = [[input_imgH // s, input_imgW // s] for s in strides]
# mapSize = [[80, 80], [40, 40], [20, 20]]
nmsThresh = 0.5
objectThresh = 0.5



class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def GenerateMeshgrid():
    for index in range(headNum):
        for i in range(mapSize[index][0]):
            for j in range(mapSize[index][1]):
                meshgrid.append(j + 0.5)
                meshgrid.append(i + 0.5)


def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS(detectResult):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + exp(-x))


def postprocess(outputs, img_h, img_w):
    """
    Generic postprocess that supports:
      - ONNX/RKNN outputs shaped (1, C, N) where C==85 (standard YOLO: 4+1+80)
      - Dual-output (two tensors) where either contains the full detection tensor
    Returns: list of DetectBox after NMS
    """
    print('postprocess ...')

    # choose a tensor to decode:
    # outputs is list of numpy arrays; each output from rknn.inference is typically (1, C, N)
    # find the first output with channels >= 6 (sane)
    chosen = None
    for out in outputs:
        arr = np.array(out)
        if arr.ndim == 3 and arr.shape[0] == 1:
            _, C, N = arr.shape
            if C >= 6:
                chosen = arr  # shape (1, C, N)
                break
    if chosen is None:
        # fallback: use first output
        chosen = np.array(outputs[0])

    # reshape to (C, N)
    chosen = chosen.reshape(chosen.shape[1], chosen.shape[2])

    C, N = chosen.shape
    # expected C == 85 for COCO-style model (4 bbox + 1 obj + num_classes)
    print(f"Decoding output with channels={C}, grid_points={N}")

    # compute grid sizes and strides (we already have mapSize & strides globals)
    # compute per-scale counts
    grid_counts = [mapSize[i][0] * mapSize[i][1] for i in range(headNum)]
    total_grid = sum(grid_counts)
    assert N == total_grid, f"Mismatch: N({N}) != total_grid({total_grid})"

    # prepare meshgrid centers (should already exist)
    # meshgrid is list [x0,y0, x1,y1, ...], length = total_grid*2
    # convert into arrays of grid_x, grid_y length N
    gx = np.array(meshgrid[0::2])   # x centers
    gy = np.array(meshgrid[1::2])   # y centers

    detect_results = []

    # If model follows standard layout:
    # per cell: [tx, ty, tw, th, obj, cls0, cls1, ...]
    if C >= (5 + class_num):
        # split channels
        tx = chosen[0, :]     # shape (N,)
        ty = chosen[1, :]
        tw = chosen[2, :]
        th = chosen[3, :]
        tobj = chosen[4, :]
        tcls = chosen[5:5 + class_num, :]  # shape (class_num, N)

        # apply sigmoid to tx,ty and objectness and class logits
        px = sigmoid_np(tx)
        py = sigmoid_np(ty)
        pobj = sigmoid_np(tobj)
        pcls = sigmoid_np(tcls)  # or softmax depending on export; sigmoid is common for multi-label/prob

        # iterate per scale to know proper stride per cell index
        start = 0
        for idx in range(headNum):
            hsize, wsize = mapSize[idx]  # e.g. [40,40]
            count = hsize * wsize
            stride = strides[idx]

            end = start + count
            # slice per-scale arrays
            sx = px[start:end]
            sy = py[start:end]
            sw = tw[start:end]
            sh = th[start:end]
            sobj = pobj[start:end]
            scls = pcls[:, start:end]  # shape (class_num, count)

            # decode boxes for this scale
            # cx = (grid_x + sx) * stride
            # cy = (grid_y + sy) * stride
            # w = exp(sw) * anchor_w  <-- if anchors used; if exported anchors included differently,
            # sometimes tw/th are already absolute widths (so you might not need exp())
            # Here we assume tw/th are log-space like: w = exp(sw) * stride (if no anchors)
            # Many models normalized w/h differently; check your original exporter. We'll attempt generic:
            # use w = np.exp(sw) * stride, h = np.exp(sh) * stride

            gx_slice = gx[start:end]
            gy_slice = gy[start:end]

            bx = (gx_slice + sx) * stride
            by = (gy_slice + sy) * stride
            bw = np.exp(sw) * stride
            bh = np.exp(sh) * stride

            # scale to original image size
            scale_w = img_w / input_imgW
            scale_h = img_h / input_imgH

            xmin = (bx - bw / 2.0) * scale_w
            ymin = (by - bh / 2.0) * scale_h
            xmax = (bx + bw / 2.0) * scale_w
            ymax = (by + bh / 2.0) * scale_h

            # clip
            xmin = np.clip(xmin, 0, img_w)
            ymin = np.clip(ymin, 0, img_h)
            xmax = np.clip(xmax, 0, img_w)
            ymax = np.clip(ymax, 0, img_h)

            # combine score = objectness * class prob (take argmax class)
            class_probs = sobj * scls  # shape (class_num, count)
            # For each cell find best class and score
            best_cls_ids = np.argmax(class_probs, axis=0)
            best_scores = class_probs[best_cls_ids, np.arange(count)]

            # collect boxes above threshold
            for i_cell in range(count):
                if best_scores[i_cell] > objectThresh:
                    cid = int(best_cls_ids[i_cell])
                    score = float(best_scores[i_cell])
                    box = DetectBox(cid, score,
                                    float(xmin[i_cell]), float(ymin[i_cell]),
                                    float(xmax[i_cell]), float(ymax[i_cell]))
                    detect_results.append(box)

            start = end

    else:
        # The output doesn't match expected 85-channel layout. Fallback: raise informative message.
        raise RuntimeError(f"Unsupported output channel count: {C}. Need 5+num_classes channels (>= {5+class_num}) or DFL layout.")

    print('detectResult:', len(detect_results))
    predBox = NMS(detect_results)
    return predBox

def export_rknn_inference(img):
    # Create RKNN object
    rknn = RKNN(verbose=False)

    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], quantized_algorithm='normal', quantized_method='channel', target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL, outputs=['output0', '1598'])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET, rknn_batch_size=1)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    rknn.release()
    print('done')

    return outputs


if __name__ == '__main__':
    print('This is main ...')
    GenerateMeshgrid()

    img_path = './test.jpg'
    orig_img = cv2.imread(img_path)
    img_h, img_w = orig_img.shape[:2]
    
    
    origimg = cv2.resize(orig_img, (input_imgW, input_imgH), interpolation=cv2.INTER_LINEAR)
    origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)
    
    img = np.expand_dims(origimg, 0)

    outputs = export_rknn_inference(img)

    out = []
    for i in range(len(outputs)):
        out.append(outputs[i])

    predbox = postprocess(out, img_h, img_w)

    print(len(predbox))

    for i in range(len(predbox)):
        xmin = int(predbox[i].xmin)
        ymin = int(predbox[i].ymin)
        xmax = int(predbox[i].xmax)
        ymax = int(predbox[i].ymax)
        classId = predbox[i].classId
        score = predbox[i].score

        cv2.rectangle(orig_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        ptext = (xmin, ymin)
        title = CLASSES[classId] + ":%.2f" % (score)
        cv2.putText(orig_img, title, ptext, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('./test_rknn_result.jpg', orig_img)
    # cv2.imshow("test", origimg)
    # cv2.waitKey(0)

