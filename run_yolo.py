#!/usr/bin/env python3

import copy
import cv2 as cv
import IPython
import math
import numpy as np
import pathlib
import PIL
import scipy.special
import tensorflow as tf
import yolo3.postprocess_np

from matplotlib import pyplot as plt

NUM_COCO_CLASSES = 80
NUM_BBOXES_PER_CELL = 3
USE_MAX_LABELS = 2
USE_MAX_BOXES = 64

THRESH_OBJNESS = 0.3
THRESH_NMS_IOU = 0.666
THRESH_CLASS = 0.1




DUMP_BLOCKS = False
DUMP_OBJNESS = False
DUMP_P_CLASS = False
DUMP_COORDS = False

DUMP_SOMETHING = DUMP_BLOCKS or DUMP_OBJNESS or DUMP_P_CLASS or DUMP_COORDS

# WHAT: #######################################################################
# different ways of decoding tiny yolo v3 output
# the best one seems to be `decode_interpreter_output_orig()`, which is an
# adaptation of the original darknet yolo decode routine.
# the yolo model is created by an ikva script:
# scripts/external_model_gen/model_gen.py  --output_dir models tiny_yolo_v3 --size 320
# #############################################################################



# keras-YOLOv3-model-set/configs/coco_classes.txt are used by default:
try:
    with open("configs/coco_classes.txt", "r", encoding="utf-8") as fil:
        class_names = [x.strip() for x in fil.readlines()]
except Exception as e:
    print(e)
    class_names = [str(n) for n in range(NUM_COCO_CLASSES)]
    class_names[0] = "human"


# from yeras-YOLOv3-model-set/common/yolo_postprocess_np.py -------------------
def yolo_decode(prediction, anchors, num_classes, input_shape):
    '''Decode final layer features to bounding box parameters.'''

    batch_size = np.shape(prediction)[0]
    assert 1 == batch_size
    num_anchors = len(anchors)

    grid_shape = np.shape(prediction)[1:3]

    # similarly, input shape seems to be not the actual NN's input shape...
    #input_shape = input_shape[1:3]

    #check if stride on height & width are same
    assert input_shape[0]//grid_shape[0] == input_shape[1]//grid_shape[1], 'model stride mismatch.'
    stride = input_shape[0] // grid_shape[0]

    prediction = np.reshape(prediction,
        (batch_size, grid_shape[0] * grid_shape[1] * num_anchors, num_classes + 5))

    # (prediction.shape) -> (1, 507(=13*13*3), 85)

    ################################
    # generate x_y_offset grid map
    grid_y = np.arange(grid_shape[0])
    grid_x = np.arange(grid_shape[1])
    x_offset, y_offset = np.meshgrid(grid_x, grid_y)

    x_offset = np.reshape(x_offset, (-1, 1))
    y_offset = np.reshape(y_offset, (-1, 1))

    x_y_offset = np.concatenate((x_offset, y_offset), axis=1)
    x_y_offset = np.tile(x_y_offset, (1, num_anchors))
    x_y_offset = np.reshape(x_y_offset, (-1, 2))
    x_y_offset = np.expand_dims(x_y_offset, 0)

    # raise ValueError(x_y_offset, x_y_offset.shape)
        #     [10, 12],
        #     [10, 12],
        #     [10, 12],
        #     [11, 12],
        #     [11, 12],
        #     [11, 12],
        #     [12, 12],
        #     [12, 12],
        #     [12, 12]]]), (1, 507, 2))

    ################################

    # Log space transform of the height and width
    anchors = np.tile(anchors, (grid_shape[0] * grid_shape[1], 1))
    anchors = np.expand_dims(anchors, 0)

    # raise ValueError(prediction.shape)  # ValueError: (1, 507, 85)

    np.set_printoptions(threshold=np.inf, suppress=True)

    box_xy = (scipy.special.expit(prediction[..., :2]) + x_y_offset) / np.array(grid_shape)[::-1]
    box_wh = (np.exp(prediction[..., 2:4]) * anchors) / np.array(input_shape)[::-1]

    # Sigmoid objectness scores
    objectness = scipy.special.expit(prediction[..., 4])  # p_o (objectness score)
    objectness = np.expand_dims(objectness, -1)  # To make the same number of values for axis 0 and 1
    #show_dist("objectness", objectness)
    # raise ValueError(objectness.shape)  # (1, 507, 1), in [0.0, 1.0]

    # Sigmoid class scores
    class_scores = scipy.special.expit(prediction[..., 5:])
    # raise ValueError(class_scores.shape)

    return np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)


def box_diou(boxes):
    """
    Calculate DIoU value of 1st box with other boxes of a box array
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    boxes: bbox numpy array, shape=(N, 4), xywh
           x,y are top left coordinates

    Returns
    -------
    diou: numpy array, shape=(N-1,)
         IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    areas = w * h

    # check IoU
    inter_xmin = np.maximum(x[1:], x[0])
    inter_ymin = np.maximum(y[1:], y[0])
    inter_xmax = np.minimum(x[1:] + w[1:], x[0] + w[0])
    inter_ymax = np.minimum(y[1:] + h[1:], y[0] + h[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)

    # box center distance
    x_center = x + w/2
    y_center = y + h/2
    center_distance = np.power(x_center[1:] - x_center[0], 2) + np.power(y_center[1:] - y_center[0], 2)

    # get enclosed area
    enclose_xmin = np.minimum(x[1:], x[0])
    enclose_ymin = np.minimum(y[1:], y[0])
    enclose_xmax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_ymax = np.maximum(x[1:] + w[1:], x[0] + w[0])
    enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
    enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
    # get enclosed diagonal distance
    enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + np.finfo(float).eps)

    return diou


def nms_boxes(boxes, classes, scores, confidence):
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        # handle data for one class
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(b)
        c_nms = copy.deepcopy(c)
        s_nms = copy.deepcopy(s)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))

            # swap the max line and first line
            b_nms[[i,0],:] = b_nms[[0,i],:]
            c_nms[[i,0]] = c_nms[[0,i]]
            s_nms[[i,0]] = s_nms[[0,i]]

            iou = box_diou(b_nms)

            # drop the last line since it has been record
            b_nms = b_nms[1:]
            c_nms = c_nms[1:]
            s_nms = s_nms[1:]

            # normal Hard-NMS
            keep_mask = np.where(iou <= THRESH_IOU)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]

    # reformat result for output
    nboxes = [np.array(nboxes)]
    nclasses = [np.array(nclasses)]
    nscores = [np.array(nscores)]
    return nboxes, nclasses, nscores



def filter_boxes(boxes, classes, scores, max_boxes):
    '''
    Sort the prediction boxes according to score
    and only pick top "max_boxes" ones
    '''
    # sort result according to scores
    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    nboxes = boxes[sorted_indices]
    nclasses = classes[sorted_indices]
    nscores = scores[sorted_indices]

    # only pick max_boxes
    nboxes = nboxes[:max_boxes]
    nclasses = nclasses[:max_boxes]
    nscores = nscores[:max_boxes]

    return nboxes, nclasses, nscores


def yolo_adjust_boxes(boxes, img_shape):
    '''
    change box format from (x,y,w,h) top left coordinate to
    (xmin,ymin,xmax,ymax) format
    '''
    if boxes is None or len(boxes) == 0:
        return []

    image_shape = np.array(img_shape, dtype='float32')
    height, width = image_shape

    adjusted_boxes = []
    for box in boxes:
        x, y, w, h = box

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
        xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
        adjusted_boxes.append([xmin,ymin,xmax,ymax])

    return np.array(adjusted_boxes,dtype=np.int32)

def yolo_handle_predictions(predictions, image_shape, num_classes, max_boxes=100, confidence=0.1, iou_threshold=THRESH_NMS_IOU, use_cluster_nms=False, use_wbf=False):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    # check if only 1 class for different score
    if num_classes == 1:
        box_scores = box_confidences
    else:
        box_scores = box_confidences * box_class_probs

    # filter boxes with score threshold
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    if use_cluster_nms:
        # use Fast/Cluster NMS for boxes postprocess
        n_boxes, n_classes, n_scores = fast_cluster_nms_boxes(boxes, classes, scores, iou_threshold, confidence=confidence)
    elif use_wbf:
        # use Weighted-Boxes-Fusion for boxes postprocess
        n_boxes, n_classes, n_scores = weighted_boxes_fusion([boxes], [classes], [scores], image_shape, weights=None, iou_thr=iou_threshold)
    else:
        # Boxes, Classes and Scores returned from NMS
        n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores, iou_threshold, confidence=confidence)

    # boxes ~ n_boxes

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes).astype('int32')
        scores = np.concatenate(n_scores)
        # from copy import copy
        # old_boxes = copy(boxes)
        boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)
        # raise ValueError(old_boxes, boxes)

        return boxes, classes, scores

    else:
        return [], [], []

# -----------------------------------------------------------------------------



def run_tflite_interpreter(tflite_model_file_bytes, test_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model_file_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    #print("INTERPRETER INPUT SHAPE: ", input_details['shape'])
    #print("TEST DATA SHAPE: ", test_data.shape)
    #input_scale = input_details["quantization_parameters"]["scales"][0]
    #input_zp = input_details["quantization_parameters"]["zero_points"][0]
    #test_data_int8 = ((test_data / input_scale) + input_zp).astype(np.int8)
    #print("INPUT DATA:\n", test_data_int8)

    interpreter.set_tensor(input_details["index"], test_data)
    interpreter.invoke()
    return interpreter


def decode_interpreter_output1(interpreter, anchors, classes, confidence=0.7,
    iou_threshold=THRESH_NMS_IOU, max_boxes=100):

    """
    original decoding from github.com:bieganski/keras-YOLOv3-model-set
    """

    num_classes = len(classes)
    assert 80 == num_classes, "expected 80 COCO classes"

    # eval.py:yolo_predict_tflite():
    output_details = interpreter.get_output_details()
    preds = [interpreter.get_tensor(outp['index']) for outp in output_details]

    # TODO: need to match the anchors to the right output, instead of this
    preds.sort(key=lambda x: len(x[0]))

    input_shape = interpreter.get_input_details()[0]['shape']
    boxes, classes, scores = yolo3.postprocess_np.yolo3_postprocess_np( \
        preds, (320,320), anchors, num_classes, (320,320))

    bboxes = []
    for n, b in enumerate(boxes):
        xmin,ymin,xmax,ymax = b

        # at this point we don't have the box confidence anymore and
        # the original box has probably been merged anyway:
        class_probs = np.zeros(num_classes)
        class_probs[classes[n]] = scores[n]
        bb = BBox(xmin, ymin, xmax, ymax, objness=1.0, class_probs=class_probs)
        bboxes.append(bb)

    return bboxes, boxes,classes,scores



def decode_interpreter_output2(interpreter, anchors, classes, confidence=0.7,
    iou_threshold=0.4, max_boxes=100):

    """
    this is an attempt to extract the post-processing from
    github.com:bieganski/keras-YOLOv3-model-set
    """

    # eval.py:yolo_predict_tflite():
    # pred_boxes, pred_classes, pred_scores = yolo3_postprocess_np(prediction,
    #    image_shape, anchors, num_classes, model_input_shape, max_boxes=100,
    #    confidence=conf_threshold, elim_grid_sense=elim_grid_sense)

    # yolo3/postprocess_np.py:yolo3_postprocess_np():
    # box_preds_concat = yolo3_decode(preds, anchors, num_classes=len(classes),
    #       input_shape=model_input_shape, elim_grid_sense=False)

    # yolo3/postprocess_np.py:yolo3_decode():

    num_classes = len(classes)
    assert 80 == num_classes, "expected 80 COCO classes"

    # eval.py:yolo_predict_tflite():
    output_details = interpreter.get_output_details()
    preds = [interpreter.get_tensor(outp['index']) for outp in output_details]

    # TODO: need to match the anchors to the right output, instead of this
    preds.sort(key=lambda x: len(x[0]))
    input_shape = interpreter.get_input_details()[0]['shape'][1:3]

    if False:
        preds = yolo3_decode(preds, anchors, num_classes=num_classes,
            input_shape=input_shape, elim_grid_sense=False)
    else:
        assert len(preds) == 2, 'prediction length does not match YOLOv3-tiny'
        assert len(preds) == len(anchors)//3, 'anchor numbers does not match prediction.'
        anchor_mask = [[3,4,5], [0,1,2]]

        box_preds = []
        for n, p in enumerate(preds):
            b = yolo_decode(p, anchors[anchor_mask[n]], num_classes, input_shape)
            box_preds.append(b)

        preds = np.concatenate(box_preds, axis=1)


    # yolo3/postprocess_np.py:yolo3_postprocess_np():
    # predictions = yolo_correct_boxes(preds, image_shape, model_input_shape)
    # "rescale prediction boxes back to original image shape"
    # TODO: ??? seems pointless. original image is at the same scale - ?
    batch_, height, width, channels_ = interpreter.get_input_details()[0]['shape']

    # TODO: also, some of this is duplicated further below...?
    box_xy = preds[..., :2]
    box_wh = preds[..., 2:4]
    objectness = np.expand_dims(preds[..., 4], -1)
    # objectness.shape -> (1, 2535, 1)
    class_scores = preds[..., 5:]

    # model_input_shape & image_shape should be (height, width) format
    model_input_shape = np.array((height, width), dtype='float32')
    image_shape = np.array((height, width), dtype='float32')

    # raise ValueError(model_input_shape/image_shape)
    new_shape = np.round(image_shape * np.min(model_input_shape/image_shape))
    # raise ValueError(image_shape, new_shape) # ValueError: (array([408., 612.], dtype=float32), array([277., 416.], dtype=float32))

    offset = (model_input_shape-new_shape)/2./model_input_shape
    # raise ValueError(model_input_shape/image_shape, offset)  # (array([1.0196079, 0.6797386], dtype=float32), array([0.1670673, 0.       ], dtype=float32))
    scale = model_input_shape/new_shape
    # raise ValueError(scale) # ValueError: [1.5018051 1.       ]
    # reverse offset/scale to match (w,h) order
    offset = offset[..., ::-1]
    # raise ValueError(offset) # ValueError: [0.        0.1670673]
    scale = scale[..., ::-1] # [1.        1.5018051]

    box_xy = (box_xy - offset) * scale
    box_wh *= scale

    # Convert centoids to top left coordinates
    box_xy -= box_wh / 2

    # Scale boxes back to original image shape.
    image_wh = image_shape[..., ::-1]
    box_xy *= image_wh
    box_wh *= image_wh

    preds = np.concatenate([box_xy, box_wh, objectness, class_scores], axis=2)
    boxes, classes, scores = yolo_handle_predictions(preds, image_shape, \
        num_classes, max_boxes=max_boxes, confidence=confidence, \
        iou_threshold=iou_threshold)
    # yolo3/postprocess_np.py:yolo3_postprocess_np():
    # boxes, classes, scores = yolo_handle_predictions(preds, image_shape,
    #    num_classes, max_boxes=max_boxes, confidence=confidence, iou_threshold=iou_threshold)
    #
    # common/yolo_postprocess_np.py:yolo_handle_predictions():
    """
    boxes = preds[:, :, :4]
    box_confs = np.expand_dims(preds[:, :, 4], -1)
    box_probs = preds[:, :, 5:]
    show_dist("box_confs", box_confs)
    box_scores = box_confs * box_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)
    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    # common/yolo_postprocess_np.py:yolo_handle_predictions():
    # n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores,
    #    iou_threshold, confidence=confidence)

    # copied (and removed in-active options) somewhere above
    print("BBBBBBBBBBBBBBB", boxes)
    n_boxes, n_classes, n_scores = nms_boxes(boxes, classes, scores,
        iou_threshold, confidence)

    if n_boxes:
        boxes = np.concatenate(n_boxes)
        classes = np.concatenate(n_classes).astype('int32')
        scores = np.concatenate(n_scores)
        boxes, classes, scores = filter_boxes(boxes, classes, scores, max_boxes)

    else:
        boxes = []
        classes = []
        scores = []
    """

    # yolo3/postprocess_np.py:yolo3_postprocess_np():
    # boxes = yolo_adjust_boxes(boxes, image_shape)

    image_shape = np.array(image_shape, dtype='float32')
    height, width = image_shape

    adjusted_boxes = []
    bboxes = []
    for n, box in enumerate(boxes):
        x, y, w, h = box

        xmin = x
        ymin = y
        xmax = x + w
        ymax = y + h

        ymin = max(0, np.floor(ymin + 0.5).astype('int32'))
        xmin = max(0, np.floor(xmin + 0.5).astype('int32'))
        ymax = min(height, np.floor(ymax + 0.5).astype('int32'))
        xmax = min(width, np.floor(xmax + 0.5).astype('int32'))
        adjusted_boxes.append([xmin,ymin,xmax,ymax])

        # at this point we don't have the box confidence anymore and
        # the original box has probably been merged anyway:
        class_probs = np.zeros(num_classes)
        class_probs[classes[n]] = scores[n]
        bb = BBox(xmin, ymin, xmax, ymax, objness=1.0, class_probs=class_probs)
        bboxes.append(bb)

    boxes = np.array(adjusted_boxes,dtype=np.int32)
    print ("BBOXES: ", bboxes)
    return bboxes, boxes, classes, scores



def decode_interpreter_output_opencv(interpreter, classes, confidence_thresh=0.1,
    nms_threshold=0.4):

    # from https://medium.com/analytics-vidhya/understanding-yolo-and-implementing-yolov3-for-object-detection-5f1f748cc63a

    # this has multiple structural problems (dimensions not matching etc),
    # doesn't look too promising for now...

    boxes = []
    class_ids = []
    confs = []

    output_details = interpreter.get_output_details()

    preds = []
    for outp in output_details:
        preds.append(interpreter.get_tensor(outp['index']))

    preds.sort(key=lambda x: len(x[0]))
    assert len(preds) == 2, 'prediction length matches YOLOv3-tiny'

    input_shape = interpreter.get_input_details()[0]['shape'][1:3]
    height, width = input_shape

    for output in preds:
        for detections in output:
            # seems wrong:
            #scores = detections[5:]
            #confidence = scores[class_id]
            """
                further up:
                box_xy = preds[..., :2]
                box_wh = preds[..., 2:4]
                objectness = np.expand_dims(preds[..., 4], -1)
                # objectness.shape -> (1, 2535, 1)
                class_scores = preds[..., 5:]
            """
            scores = detections[..., 5:]
            class_id = np.argmax(scores, axis=-1)
            confidence = np.max(scores, axis=-1)
            #IPython.embed()

            if confidence > confidence_thresh:
                #centre_x = int(detections[0] * width)
                #centre_y = int(detections[1] * height)
                # ^ original seems the wrong way around, no?
                centre_y = int(detections[0] * height)
                centre_x = int(detections[1] * width)
                h = int(detection[2] * height)
                w = int(detection[3] * width)

                x = int(centre_x - (w/2))
                y = int(centre_y - (h/2))

                boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)
    colours = np.random.uniform(0, 255, size=(len(boxes), 3))

    for n in indices.flatten():
        x, y, w, h = boxes[n]
        label = str(classes[class_ids[n]])
        confidence = str(round(confs[n], 2))
        colour = colours[n]

        #cv.rectangle(img, (x, y), (x+w, y+h), colour, 2)
        #cv.putText(img, f"{label} {confidence}", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6,
        #   (255,255,255), 2)

    #cv.imshow(image_file, img)
    #key = cv.waitKey(1)
    return boxes, classes, confs



def clamp(val, min_=0.0, max_=1.0):
    return min(max_, max(min_, val))


class BBox:
    """
    coords must already been scaled to image size
    """

    def __init__(self, xmin, ymin, xmax, ymax, objness=None, class_probs=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.class_probs = class_probs
        self.label = -1
        self.score = -1
        self.show_as = None


    def get_label(self):
        if self.label == -1:
            self.label = class_names[np.argmax(self.class_probs)]

        return self.label


    def get_score(self):
        if self.score == -1:
            self.score = np.max(self.class_probs)

        return self.score


    def __repr__(self):
        class_probs = ", ".join([f"{(100*c[0]):.1f}% >{c[1]}<" for c in self.guess_class()])
        return f"obj: {self.objness:.3f}, {class_probs} @ ({self.xmin:.3f}, {self.ymin:.3f}) / ({self.xmax:.3f}, {self.ymax:.3f})"


    def __str__(self):
        return self.__repr__()


    def guess_class(self):
        if self.show_as is None:
            possibilities = [(self.class_probs[n], l) for n, l in enumerate(class_names)]
            possibilities.sort(key = lambda x: x[0], reverse=True)
            possibilities = [p for p in possibilities if p[0] > 1e-6]
            self.show_as = possibilities[:USE_MAX_LABELS]

        return self.show_as




"""
def _sigmoid2(x):
    # slightly modified.. doesn't overflow, but doesn't work on vectors/np.arrays..
    if x >= 0:
        z = np.exp(-x)
    else:
        z = np.exp(x)
    return 1. / (1. + z)
"""

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def show_dist(s, x):
    counts, vals = np.histogram(x)
    print(s)
    print("\tcount\tbin")
    for z in zip(counts, vals):
        print(f"\t{z[0]}\t{z[1]:.1f}")


def summarise_boxes(boxes):
    if 0 == len(boxes):
        print("no boxes found.")
        return

    all_objness = [x.objness for x in boxes]
    all_highest_class = [max(x.class_probs) for x in boxes]

    print(f"{len(boxes)} boxes, highest objness: {max(all_objness):.2f}, "\
        f"highest class confidence: {max(all_highest_class):.2f}")
    show_dist("objness distribution:", all_objness)
    show_dist("highest class confidence distribution:", all_highest_class)


def decode_interpreter_output_orig(interpreter, anchors, \
    obj_thresh=0.4, nms_thresh=0.6, class_thresh=0.2, \
    use_max_boxes=USE_MAX_BOXES, use_max_labels=USE_MAX_LABELS):

    """
    following the original darknet yolo decoding with changes suggested by
    cedric and tom (soft max)
    """

    SOFT_MAX = True

    output_details = interpreter.get_output_details()
    input_shape = interpreter.get_input_details()[0]['shape'][1:3]
    net_h, net_w = input_shape
    assert net_h == net_w, "the input image is supposed to be a square"

    boxes = []
    for outp in output_details:
        outp_name = outp['name']
        outp_anchors = anchors[outp_name]
        p = interpreter.get_tensor(outp['index'])[0]
        grid_h, grid_w = p.shape[0:2]
        assert grid_h == grid_w, "the net input is supposed to be square"
        assert (net_h / grid_h) == (net_w / grid_w), "output shape seems weird"

        netout = p.reshape((grid_h, grid_w, NUM_BBOXES_PER_CELL, -1))
        assert netout.shape[-1] - 5 == len(class_names)

        # we use a de-quantisation layer in the model now
        # but later we've got to deal with the int8 output.
        if False:
            quant_params = outp['quantization_parameters']
            q_scale = quant_params['scales'][0]
            q_off = quant_params['zero_points'][0]
            netout_pre_quant = netout_i8.astype(float)
            netout = q_scale * (netout_pre_quant - q_off)
        else:
            netout = netout

        #dump_array(f"net_output-{grid_w}x{grid_h}.h", "float", netout)
        # class_probs   *= corresponding bbox's objness:
        # see below - first softmax on these, then multiply..
        # shape (13, 13, 3, 80)  *=  shape (13, 13, 3, 1)
        # netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]

        if DUMP_BLOCKS:
            with open(f"blocks_{grid_w}x{grid_h}", "w") as outf:
                re = netout.reshape(-1, 85)
                print("[", file=outf)
                for b in range(grid_w*grid_h*NUM_BBOXES_PER_CELL):
                    print("[", file=outf, end="")
                    for x in range(85):
                        print(f"{re[b][x]}, ", end="", file=outf)
                    print("],", file=outf)
                print("]", file=outf)

        objness_orig = netout[..., 4].flatten()
        show_dist("objness", objness_orig)
        netout[..., 4] = scipy.special.expit(netout[..., 4])
        objness_sig = netout[..., 4].flatten()
        show_dist("objness_sig", objness_sig)

        if DUMP_OBJNESS:
            with open(f"objness_{grid_w}x{grid_h}", "w") as outf:
                print(f"py_np_{grid_w}_raw = [" + ", ".join([str(x) for x in objness_orig]) + "]", file=outf)
                print(f"py_np_{grid_w}_sig = [" + ", ".join([str(x) for x in objness_sig]) + "]", file=outf)

        if DUMP_P_CLASS:
            outf = open(f"p_class_{grid_w}x{grid_h}", "w")
            print(f"py_np_{grid_w} = np.array([", file=outf)

        if DUMP_COORDS:
            outf = open(f"coords_{grid_w}x{grid_h}", "w")
            print(f"py_np_{grid_w} = np.array([", file=outf)
            print("\n\n")

        for n in range(grid_h * grid_w):
            row = int(n / grid_w)
            col = int(n % grid_w)

            for b in range(NUM_BBOXES_PER_CELL):
                objness = netout[row][col][b][4]

                if objness <= obj_thresh: continue

                class_probs = netout[row][col][b][5:]

                if DUMP_P_CLASS:
                    print("[[" + ", ".join([str(x) for x in class_probs]) + "],", file=outf)

                if SOFT_MAX:
                    class_probs = objness * tf.nn.softmax(class_probs)
                else:
                    class_probs = objness * class_probs

                if DUMP_P_CLASS:
                    print("[" + ", ".join([str(float(x)) for x in class_probs]) + "]],\n", file=outf)

                """
                darknet/src/yolo_layer.c:
                 88     b.x = (i + x[index + 0*stride]) / lw;
                 89     b.y = (j + x[index + 1*stride]) / lh;
                 90     b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
                 91     b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;

                tf android yolo demo:
                # final float xPos = (x + expit(output[offset + 0])) * blockSize;
                """

                x, y, w, h = netout[row, col, b, 0:4]
                x, y = scipy.special.expit((x, y))

                x = (col+x)/grid_w
                y = (row+y)/grid_h

                w = np.exp(w) * outp_anchors[b][1] / net_w
                h = np.exp(h) * outp_anchors[b][0] / net_h

                if DUMP_COORDS:
                    print(f"[{x}, {y}, {w}, {h}],", file=outf)

                box = BBox(x - w/2, y - h/2, x + w/2, y + h/2, objness, np.array(class_probs))
                boxes.append(box)

        if DUMP_P_CLASS or DUMP_COORDS:
            print("])", file=outf)
            outf.close()


    #print("\nFOUND:", len(boxes), " boxes: ", boxes)
    #summarise_boxes(boxes)
    do_nms(boxes, nms_thresh)
    boxes.sort(key = lambda b: b.objness, reverse=True)
    boxes = [b for b in boxes if (b.class_probs > class_thresh).any()]
    #print(f"\nMERGED via NMS into:")
    #summarise_boxes(boxes)
    return boxes



def decode_interpreter_output_kaggle(interpreter, anchors, \
    obj_thresh=0.6, nms_thresh=0.5, class_thresh=0.2, \
    use_max_boxes=USE_MAX_BOXES, use_max_labels=USE_MAX_LABELS):

    # https://www.kaggle.com/code/ankitp013/step-by-step-yolov3-object-detection
    output_details = interpreter.get_output_details()

    input_shape = interpreter.get_input_details()[0]['shape'][1:3]
    net_h, net_w = input_shape

    boxes = []
    for outp in output_details:
        outp_name = outp['name']
        outp_anchors = anchors[outp_name]
        p = interpreter.get_tensor(outp['index'])[0]

        #print("GRID ", p.shape)
        #print("ANCHORS: ", outp_anchors
        # boxes += decode_netout(p, ....)
        grid_h, grid_w = p.shape[0:2]
        netout_i8 = p.reshape((grid_h, grid_w, NUM_BBOXES_PER_CELL, -1))
        # shape is now (gridy, gridx, bbox, bbox_props), e. g. 13x13x3x85
        #       (where bbox_props are 80 classes + x,y,w,h,conf)
        assert netout_i8.shape[-1] - 5 == len(labels)

        #print("SCALE ", scale)
        #show_dist("x_int8", netout_i8[...,0].flatten())
        #show_dist("y_int8", netout_i8[...,1].flatten())
        quant_params = outp['quantization_parameters']
        q_scale = quant_params['scales'][0]
        q_off = quant_params['zero_points'][0]
        netout = netout_i8.astype(float)

        """output quantisation??:
          0 : quantization: 0.055902499705553055 * (q - 18)
          1 : quantization: 0.04634712636470795 * (q - 1)
        """

        #netout[..., 0:2] = _sigmoid(netout[..., 0:2])   # x, y
        netout[..., 0:2] = scipy.special.expit(netout[..., 0:2])   # x, y

        # ??? shouldn't there be a sigmoid on the w/h as well?
        #netout[..., 0:4] = scipy.special.expit(netout[..., 0:4])   # x, y, w, h

        netout[..., 4:] = scipy.special.expit(netout[..., 4:])     # objness, class_probs

        # class_probs   *= corresponding bbox's objness:
        # shape (13, 13, 3, 80)  *=  shape (13, 13, 3, 1)
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]

        # this seems broken:
        # filter out all class probs below threshold:
        # netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for n in range(grid_h * grid_w):
            row = int(n / grid_w)
            col = int(n % grid_w)

            for b in range(NUM_BBOXES_PER_CELL):
                objness = netout[row][col][b][4]
                class_probs = netout[row][col][b][5:]

                if objness <= obj_thresh: continue
                #print("P_class: ", class_probs)

                x, y, w, h = netout[row][col][b][0:4]

                if True:
                    # coords rel to cell -> bbox centre
                    x = (col + x) / grid_w * net_w
                    y = (row + y) / grid_h * net_h

                    # box size is log scale
                    w = anchors[scale][b][0] * np.exp(w)
                    h = anchors[scale][b][1] * np.exp(h)

                else:
                    # original. coords seem wrong, also want image coords anyway
                    # coords rel to cell -> bbox centre, normalise to image size
                    x = (col + x) / grid_w
                    y = (row + y) / grid_h

                    # box size is log scale, normalise to image size
                    w = anchors[scale][b][0] * np.exp(w) / net_w
                    h = anchors[scale][b][1] * np.exp(h) / net_h


                box = BBox(x-w/2, y-h/2, x+w/2, y+h/2, objness, class_probs)
                boxes.append(box)

    print("\nFOUND:")
    summarise_boxes(boxes)
    do_nms(boxes, nms_thresh, class_thresh)
    boxes.sort(key = lambda b: b.objness, reverse=True)
    print(f"\nMERGED via NMS into:")
    summarise_boxes(boxes)
    print(boxes)
    return boxes[:USE_MAX_BOXES]



# also from https://www.kaggle.com/code/ankitp013/step-by-step-yolov3-object-detection
# shouldn't be needed as img has same size as NN input - ???
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)



def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    if union == 0.0: return 0.0
    return float(intersect) / union

def do_nms(boxes, nms_thresh, class_thresh=0.0):
    if len(boxes) > 0:
        nb_class = len(boxes[0].class_probs)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.class_probs[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].class_probs[c] <= class_thresh: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].class_probs[c] = 0



def draw_boxes(filename, boxes, scale=False, ref_boxes=None):
    data = plt.imread(filename)
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    img_width, img_height, _ = data.shape

    def dbox(boxes, colour):
        for box in boxes:
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
            width, height = x2 - x1, y2 - y1

            # coords and size are normalised to image size
            if scale:
                x1, x2, width = img_width * np.array([x1, x2, width])
                y1, y2, height = img_height * np.array([y1, y2, height])

            #print(f"BBBBB x,y,w,h: {x1}, {y1}, {width}, {height}")
            rect = plt.Rectangle((x1, y1), width, height, fill=False, color=colour)
            ax.add_patch(rect)

            # draw text and score in top left corner
            # show multiple interpretations, if they are close
            thresh = .8 * box.guess_class()[0][0]
            what = [f"{x[1]} ({int(100*x[0])}%)" for x in box.guess_class() if x[0] > thresh]
            what = what[:USE_MAX_LABELS]
            label = ", ".join(what)
            plt.text(x1, y1, label, backgroundcolor=colour)

    dbox(boxes, 'red')
    if ref_boxes: dbox(ref_boxes, 'green')
    plt.show()


def pre_process_image(fname):
    img = PIL.Image.open(fname)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = (np.array(img, dtype=np.float32) - 128.0).astype(np.int8)

    # 0-th dimension is the batch index (about which we don't care here, but the model
    # needs the data in the correct dimensions)
    img_array = np.expand_dims(img_array, 0)
    return img_array.astype(np.int8)


def dump_array(outf_name: str, data_type: str, np_array):
    with open(outf_name, "w") as outf:
        print("#ifndef __NETWORK_INPUT_H__", file=outf)
        print("#define __NETWORK_INPUT_H__", file=outf)
        print("#include <cstdint>", file=outf)

        np_array = np_array.flatten()
        print(f"{data_type} network_input[{len(np_array)}] = {{", file=outf)

        off_last = 8 * math.ceil(len(np_array) / 8)

        off=0
        while off < off_last:
            print("\t" + ", ".join([str(x) for x in np_array[off:off+8]]) + ",", file=outf)
            off += 8

        print("};", file=outf)
        print("#endif", file=outf)

def human_box(p_obj_pct, xmin, ymin, xmax, ymax, p_human_pct):
    p_class = np.array([p_human_pct/100.0] + ([0] * (NUM_COCO_CLASSES-1)))
    bb = BBox(xmin, ymin, xmax, ymax, objness=p_obj_pct/100.0, class_probs=p_class)
    return bb

# "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.int8.tflite-2023-01-31.tflite"
ref_boxes = [
    human_box(58.7, 0.00, 0.36, 0.31, 0.78, 58.7),
    human_box(37.1, 0.16, 0.34, 0.53, 0.77, 37.1),
    human_box(93.4, 0.29, 0.34, 0.60, 0.77, 93.4),
    human_box(70.7, 0.41, 0.33, 0.72, 0.75, 70.7),
    human_box(85.3, 0.47, 0.31, 0.84, 0.74, 85.3),
    human_box(33.1, 0.52, 0.34, 0.96, 0.76, 33.1),
    human_box(97.6, 0.70, 0.36, 1.01, 0.78, 97.6),
]

# "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.int8.tflite-2023-02-15.tflite"
ref_boxes = [
    human_box(0.976*100, 0.699, 0.356, 1.010, 0.778, 97.6),
    human_box(0.934*100, 0.286, 0.343, 0.597, 0.765, 93.4),
    human_box(0.853*100, 0.473, 0.315, 0.844, 0.737, 85.3),
    human_box(0.707*100, 0.407, 0.330, 0.719, 0.752, 70.7),
    human_box(0.587*100, 0.003, 0.356, 0.314, 0.778, 58.7),
]

ref_boxes = None


if "__main__" == __name__:
    # model file is generated by stuff in the keras-YOLOv3-model-set repo, see notes
    #model_file = "tiny_yolo_v3_416.tflite"
    #model_file = "tiny_yolo_v3size320.int8.tflite"
    model_file = "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.int8.tflite-2023-02-15.tflite"

    # defined in the keras-YOLOv3-model-set repo, duplicated here for brevity
    # supposed to be used with   26x26                   13x13 ???
    anchors = np.array(((10,14), (23,27), (37,58), (81,82), (135,169), (344,319)))

    # image file dimension *must* *match* the model (input dimensions)!!
    #image_file = "test_images/example_image-320x320.bmp"
    image_file = "test_images/example_image-320x320.jpeg"
    #image_file = "test_images/muppets-1.jpeg"
    #image_file = "test_images/horses.jpg"
    #image_file = "test_images/dog.jpg"

    model_bytes = pathlib.Path(model_file).read_bytes()
    img_array = pre_process_image(image_file)
    #dump_array("network_input.h", "int8_t", img_array)
    interpreter = run_tflite_interpreter(model_bytes, img_array)


    # both seem quite broken:
    #boxes_cv, classes_cv, scores_cv = decode_interpreter_output_opencv(interpreter, anchors, class_names)
    #bboxes = decode_interpreter_output_kaggle(interpreter, anchors2)

    if False:
        scale = False
        # original decoding routines from this repo
        bboxes, boxes, classes, scores = decode_interpreter_output1(interpreter, anchors, class_names)
        # same, but trimmed down....
        # this performs much worse than the previous, for some reason. sigmoid missing?
        #bboxes, boxes, classes, scores = decode_interpreter_output2(interpreter, anchors, class_names)

    else:
        anchors2 = {
            'StatefulPartitionedCall:0' : np.array([(10,14), (23,27), (37,58)]),
            'StatefulPartitionedCall:1' : np.array([(81,82), (135,169), (344,319)])
        }

        bboxes = decode_interpreter_output_orig(interpreter, anchors2)
        scale=True


    bboxes = bboxes[:USE_MAX_BOXES]
    summarise_boxes(bboxes)

    for b in bboxes:
        print(b)

    if len(bboxes) > 0:
        draw_boxes(image_file, bboxes, scale, ref_boxes=ref_boxes)