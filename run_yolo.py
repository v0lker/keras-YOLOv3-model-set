#!/usr/bin/env python3

# WHAT: #######################################################################
# the yolo model is created by an ikva script:
# scripts/external_model_gen/model_gen.py  --output_dir models tiny_yolo_v3 --size 320
# #############################################################################


import copy
import IPython
import lce_utils
import math
import numpy as np
import pathlib
import PIL
import scipy.special
import tensorflow as tf
import yolo3.postprocess_np

from dataclasses import dataclass
from matplotlib import pyplot as plt
from typing import List

DEQUANT = True
FORCE_IKVA_INTERPRETER = True

NET_IN_SZ = 320   # image has to be NET_IN_SZ x NET_IN_SZ
NUM_COCO_CLASSES = 80
NUM_VALS_PER_BOX = NUM_COCO_CLASSES + 5  # 4 coords + objness + class probs
NUM_BBOXES_PER_CELL = 3
USE_MAX_LABELS = 2
USE_MAX_BOXES = 64

THRESH_OBJNESS = 0.3
THRESH_NMS_IOU = 0.666
THRESH_CLASS = 0.1

DUMP_RAW = False
DUMP_BLOCKS = True
DUMP_OBJNESS = False
DUMP_P_CLASS = False
DUMP_COORDS = False

DUMP_SOMETHING = DUMP_RAW or DUMP_BLOCKS or DUMP_OBJNESS or DUMP_P_CLASS or DUMP_COORDS



# keras-YOLOv3-model-set/configs/coco_classes.txt are used by default:
try:
    with open("configs/coco_classes.txt", "r", encoding="utf-8") as fil:
        class_names = [x.strip() for x in fil.readlines()]
except Exception as e:
    print(e)
    class_names = [str(n) for n in range(NUM_COCO_CLASSES)]
    class_names[0] = "human"


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
        return f"obj: {100*self.objness:.1f}%, {class_probs} @ ({self.xmin:.3f}, {self.ymin:.3f}) / ({self.xmax:.3f}, {self.ymax:.3f})"


    def __str__(self):
        return self.__repr__()


    def guess_class(self):
        if self.show_as is None:
            possibilities = [(self.class_probs[n], l) for n, l in enumerate(class_names)]
            possibilities.sort(key = lambda x: x[0], reverse=True)
            possibilities = [p for p in possibilities if p[0] > 1e-6]
            self.show_as = possibilities[:USE_MAX_LABELS]

        return self.show_as



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




@dataclass
class InterpreterOutput:
    name: str
    input_shape: List[int]
    output_shape: List[int]
    anchors: np.array
    data: np.array
    quant_off: float
    quant_scale: float




def get_interpreter_output(model_file, test_data, anchors):
    """
    returns: list of InterpreterOutputs

    """

    is_ikva = "ikva" in model_file
    print(f"\n>>>>>>>>> USING{' IKVA' * is_ikva} MODEL {model_file}\n")

    model_bytes = pathlib.Path(model_file).read_bytes()

    ret = []

    if FORCE_IKVA_INTERPRETER or is_ikva:
        model_ikva = lce_utils.memory_planner.ikva_memory_planner.plan_ikva_memory(
            model_bytes, pe_mem_size=16*1024, num_families=1, word_size=8)

        interpreter = lce_utils.LCEIkvaInterpreter(model_ikva,
            arena_size_bytes = 4*1024*1024, strict_ikva_ops=False)

        input_shape = interpreter.input_shapes[0][1:3]
        netin_h, netin_w = input_shape
        outputs = interpreter.predict(test_data)

        outp_name = { 20: 'StatefulPartitionedCall:0', 10: 'StatefulPartitionedCall:1' }

        for n, outp in enumerate(outputs):
            netout_w, netout_h = outp.shape[1:3]
            name = outp_name[netout_w]

            ret.append(InterpreterOutput(
                name = name,
                input_shape = [netin_w, netin_h],
                output_shape = [netout_w, netout_h],
                anchors = anchors[name],
                data = outp,
                quant_off = interpreter.output_zero_points[n],
                quant_scale = interpreter.output_scales[n]
            ))

    else:
        interpreter = tf.lite.Interpreter(model_content=model_bytes)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        #print("INTERPRETER INPUT SHAPE: ", input_details['shape'])
        #print("TEST DATA SHAPE: ", test_data.shape)
        #input_scale = input_details["quantization_parameters"]["scales"][0]
        #input_zp = input_details["quantization_parameters"]["zero_points"][0]
        #test_data_int8 = ((test_data / input_scale) + input_zp).astype(np.int8)
        #print("INPUT DATA:\n", test_data_int8)
        interpreter.set_tensor(input_details["index"], test_data)
        input_shape = interpreter.get_input_details()[0]['shape'][1:3]
        netin_h, netin_w = input_shape

        interpreter.invoke()
        output_details = interpreter.get_output_details()

        for outp in output_details:
            grid_h, grid_w = outp['shape'][1:3]
            netout = interpreter.get_tensor(outp['index'])[0]
            quant_params = outp['quantization_parameters']
            q_scale = quant_params['scales'][0]
            q_off = quant_params['zero_points'][0]

            ret.append(InterpreterOutput(
                name = outp['name'],
                input_shape = [netin_w, netin_h],
                output_shape = [grid_w, grid_h],
                anchors = anchors[outp['name']],
                data = netout,
                quant_off = q_off,
                quant_scale = q_scale
            ))

    assert netin_h == netin_w, "the input image is supposed to be a square"

    for outp in ret:
        netout_w, netout_h = outp.output_shape

        assert netout_h == netout_w, f"the net output is supposed to be square but have {netout_h}x{netout_w}"
        assert (netin_h / netout_h) == (netin_w / netout_w), "same aspect ratio for in/out"

        netout = outp.data.reshape((netout_h, netout_w, NUM_BBOXES_PER_CELL, -1))
        assert netout.shape[-1] - 5 == len(class_names)

        if DUMP_RAW:
            re = netout.reshape(-1, NUM_VALS_PER_BOX)
            print(f"raw_py_{netout_w}x{netout_h} = np.array([", file=outf)
            for b in range(netout_w*netout_h*NUM_BBOXES_PER_CELL):
                print("[", file=outf, end="")
                for x in range(NUM_VALS_PER_BOX):
                    print(f"{re[b][x]}, ", end="", file=outf)
                print("],", file=outf)
            print("])\n", file=outf)

    return ret



def make_blocks(output_data):
    for outp in output_data:
        netin_w, netin_h = outp.input_shape
        netout_w, netout_h = outp.output_shape
        netout_pre_quant = outp.data.astype(np.float32)

        if DEQUANT:
            netout = outp.quant_scale * (netout_pre_quant - outp.quant_off)
            #print("GRID: ", grid_h, "; SCALE: ", q_scale, "; OFF: ", q_off)
        else:
            netout = netout_pre_quant

        netout = netout.reshape((netout_w, netout_h, NUM_BBOXES_PER_CELL, -1))

        if DUMP_BLOCKS:
            dump_out = netout.reshape(-1, NUM_VALS_PER_BOX)
            print(f"blocks_{netout_w}x{netout_h} = np.array([", file=outf)
            for b in range(netout_w * netout_h * NUM_BBOXES_PER_CELL):
                print("[" + ", ".join([str(x) for x in dump_out[b]]) + "]\n", file=outf)
            print("])\n", file=outf)

        outp.data = netout



def decode_yolo(output_data):
    boxes = []

    for outp in output_data:
        netin_w, netin_h = outp.input_shape
        netout_w, netout_h = outp.output_shape

        netout = outp.data.reshape((netout_w, netout_h, NUM_BBOXES_PER_CELL, -1))
        objness_orig = netout[..., 4].flatten()
        objness_sig = scipy.special.expit(netout[..., 4]).flatten()

        if DUMP_OBJNESS:
            print(f"py_objness_{outp.output_shape}_raw = np.array([" \
                + ", ".join([str(x) for x in objness_orig]) + "])\n", file=outf)

            print(f"py_objness_{outp.output_shape}_sig = np.array([" \
                + ", ".join([str(x) for x in objness_sig]) + "])\n", file=outf)

        if DUMP_P_CLASS:
            print(f"py_p_class_{outp.output_shape} = np.array([", file=outf)

        if DUMP_COORDS:
            print(f"py_coords_{outp.output_shape} = np.array([", file=outf)
            print("\n\n")

        for n in range(netout_w * netout_h):
            row = int(n / netout_w)
            col = int(n % netout_w)

            for b in range(NUM_BBOXES_PER_CELL):
                objness = _sigmoid(netout[row][col][b][4])

                if objness <= THRESH_OBJNESS: continue

                class_probs = netout[row][col][b][5:]

                # TODO: do these still work??
                if DUMP_P_CLASS:
                    print("[[" + ", ".join([str(x) for x in class_probs]) + "],", file=outf)

                class_probs = objness * tf.nn.softmax(class_probs)

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
                x = _sigmoid(x)
                y = _sigmoid(y)

                x = (col+x)/netout_w
                y = (row+y)/netout_h

                w = np.exp(w) * outp.anchors[b][1] / netin_w
                h = np.exp(h) * outp.anchors[b][0] / netin_h

                if DUMP_COORDS:
                    print(f"[{x}, {y}, {w}, {h}],", file=outf)

                box = BBox(x - w/2, y - h/2, x + w/2, y + h/2, objness, np.array(class_probs))
                boxes.append(box)

                if DUMP_P_CLASS or DUMP_COORDS:
                    print("])\n", file=outf)


    #print("\nFOUND:", len(boxes), " boxes: ", boxes)
    #summarise_boxes(boxes)
    do_nms(boxes)
    boxes.sort(key = lambda b: b.objness, reverse=True)
    boxes = [b for b in boxes if (b.class_probs > THRESH_CLASS).any()]
    #print(f"\nMERGED via NMS into:")
    #summarise_boxes(boxes)
    return boxes



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

def do_nms(boxes):
    if len(boxes) > 0:
        nb_class = len(boxes[0].class_probs)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.class_probs[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].class_probs[c] <= THRESH_CLASS: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= THRESH_NMS_IOU:
                    boxes[index_j].class_probs[c] = 0



def draw_boxes(filename, boxes, scale=False, ref_boxes=None):
    data = plt.imread(filename)
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    img_width, img_height, _ = data.shape

    def dbox(boxes, colour):
        for n, box in enumerate(boxes):
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
            label = f"{n}: " + ", ".join(what)

            if "red" == colour:
                yl = y1 + height/2
            else:
                yl = y1

            plt.text(x1, yl, label, backgroundcolor=colour)

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


def cmp_boxes(dut, ref):
    assert ref, "ref"
    assert len(ref)>0, "ref > 0"

    if not dut or (len(dut) <= 0):
        print("test data empty")
        return

    dut.sort(key = lambda x: x.objness, reverse=True)
    ref.sort(key = lambda x: x.objness, reverse=True)

    if len(dut) == len(ref):
        print("length matches")
    else:
        print(f"length: {len(dut)}, expected: {len(ref)}")

    m_iou = []     # matrix_dut_ref of IOU(dut, ref)
    m_obj = []     #                   abs(diff(objness))
    m_pcl = []     #                   sum(abs(diff(p_classes)))

    for b in dut:
        ious = [bbox_iou(b, r) for r in ref]
        m_iou.append(ious)

        objs = [abs(b.objness - r.objness) for r in ref]
        m_obj.append(objs)

        pcls = [sum(abs(b.class_probs - r.class_probs)) for r in ref]
        m_pcl.append(pcls)

    def print_some_boxes(name, boxes):
        print(f"\n{name}:")
        for n, b in enumerate(boxes):
            print(f"{n}: {b}")

    print_some_boxes("TESTEE", dut)
    print_some_boxes("REF", ref)

    def print_one_matrix(name, m):
        print(f"\n{name}:")
        print("\t" + "\t".join([str(x) for x in range(len(m[0]))]))
        for n, it in enumerate(m):
            print(f"{n}\t" + "\t".join([f"{int(100*x)}" for x in it]))

    print_one_matrix("IOU", m_iou)
    print_one_matrix("abs objness error", m_obj)
    print_one_matrix("abs class error", m_pcl)



# 7 boxes from int8 in gstreamer demo and this script here:
# "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.int8.tflite-2023-01-31.tflite"
# "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.int8.tflite-2023-02-15.tflite"
# human_box(p_obj_pct, xmin, ymin, xmax, ymax, p_human_pct)
# this is also the result of tiny_yolo_v3size320.int8_zero_restricted.tflite *on x86*
# (but currently not on briey nor ikva)
ref_boxes = [
    human_box(97.6, 0.70, 0.36, 1.01, 0.78, 97.6),
    human_box(93.4, 0.29, 0.34, 0.60, 0.77, 93.4),
    human_box(85.3, 0.47, 0.31, 0.84, 0.74, 85.3),
    human_box(70.7, 0.41, 0.33, 0.72, 0.75, 70.7),
    human_box(58.7, 0.00, 0.36, 0.31, 0.78, 58.7),
    human_box(37.1, 0.16, 0.34, 0.53, 0.77, 37.1),
    human_box(33.1, 0.52, 0.34, 0.96, 0.76, 33.1),
]

# 11 boxes from int8 model on briey
#ref_boxes = [
#    human_box(41, 0.45, 0.21, 0.67, 0.47, 41),
#    human_box(41, -0.2, 0.28, 0.19, 0.78, 41),
#    human_box(33, 0.5,  0.35, 0.27, 0.78, 33),
#    human_box(58, 0.0,  0.35, 0.31, 0.77, 58),
#    human_box(45, 0.5,  0.35, 0.42, 0.77, 45),
#    human_box(45, 0.18, 0.34, 0.49, 0.76, 45),
#    human_box(95, 0.30, 0.34, 0.56, 0.76, 95),
#    human_box(80, 0.43, 0.32, 0.69, 0.74, 80),
#    human_box(90, 0.47, 0.31, 0.84, 0.73, 90),
#    human_box(50, 0.51, 0.33, 0.95, 0.75, 49),
#    human_box(98, 0.70, 0.35, 1.1,  0.78, 98),
#]

# 11 boxes from int8 (implicit de-quantisation) model on briey with less rounding:
#NOref_boxes = [
#    human_box(100*0.412793, 0.459502, 0.219043, 0.674351 , 0.472168, 100*0.412792),
#    human_box(100*0.412793, -0.026406, 0.285499, 0.192415, 0.788668, 100*0.412771),
#    human_box(100*0.330735, 0.055492, 0.359487, 0.270341 , 0.788948, 100*0.330718),
#    human_box(100*0.587207, -0.001245, 0.351979, 0.310033, 0.773854, 100*0.587203),
#    human_box(100*0.456060, 0.055650, 0.351979, 0.426909 , 0.773854, 100*0.456060),
#    human_box(100*0.456060, 0.185641, 0.343457, 0.496918 , 0.765332, 100*0.456060),
#    human_box(100*0.959763, 0.306590, 0.343457, 0.567576 , 0.765332, 100*0.959763),
#    human_box(100*0.803723, 0.436433, 0.326146, 0.697420 , 0.748021, 100*0.803722),
#    human_box(100*0.908113, 0.477287, 0.314845, 0.848546 , 0.736720, 100*0.908113),
#    human_box(100*0.500000, 0.511674, 0.334668, 0.954473 , 0.756543, 100*0.500000),
#    human_box(100*0.987936, 0.703082, 0.359767, 1.014359 , 0.781642, 100*0.987936),
#]

# 11 boxes from int8 (explicit de-quantisation, zero point at 0) on briey
ref_boxes = [
    human_box(100*0.456060, -0.026406,0.285499, 0.192415, 0.788668, 0.456033),
    human_box(100*0.330735, 0.059502, 0.359487, 0.274351, 0.788948, 0.330722),
    human_box(100*0.669266, 0.003082, 0.351979, 0.314359, 0.773854, 0.669263),
    human_box(100*0.412793, 0.059976, 0.347783, 0.431236, 0.769658, 0.412793),
    human_box(100*0.412793, 0.181445, 0.343457, 0.492722, 0.765332, 0.412793),
    human_box(100*0.330735, 0.347014, 0.244540, 0.527152, 0.855460, 0.330706),
    human_box(100*0.952378, 0.306590, 0.347783, 0.567576, 0.769658, 0.952378),
    human_box(100*0.830045, 0.443724, 0.326146, 0.704711, 0.748021, 0.830044),
    human_box(100*0.892314, 0.468764, 0.314845, 0.840024, 0.736720, 0.892313),
    human_box(100*0.456060, 0.507896, 0.334668, 0.950695, 0.756543, 0.456059),
    human_box(100*0.985645, 0.707278, 0.359767, 1.018555, 0.781642, 0.985645),
]



show_boxes = []

# 11 boxes from: int8 model (implicit de-quantisation) on briey net out -> py ref post-processing:
show_boxes = [
    human_box(98.8, 0.703, 0.360, 1.014, 0.782, 98.8),
    human_box(96.0, 0.307, 0.343, 0.568, 0.765, 96.0),
    human_box(90.8, 0.477, 0.315, 0.849, 0.737, 90.8),
    human_box(80.4, 0.436, 0.326, 0.697, 0.748, 80.4),
    human_box(58.7, -0.001, 0.352, 0.310, 0.774,58.7),
    human_box(50.0, 0.512, 0.335, 0.954, 0.757, 50.0),
    human_box(45.6, 0.056, 0.352, 0.427, 0.774, 45.6),
    human_box(45.6, 0.186, 0.343, 0.497, 0.765, 45.6),
    human_box(41.3, 0.460, 0.219, 0.674, 0.472, 41.3),
    human_box(41.3, -0.026, 0.285, 0.192, 0.789,41.3),
    human_box(33.1, 0.055, 0.359, 0.270, 0.789, 33.1),
]





if "__main__" == __name__:
    # image file dimension *must* *match* the model (input dimensions)!!
    #image_file = "test_images/example_image-320x320.bmp"
    image_file = "test_images/example_image-320x320.jpeg"
    #image_file = "test_images/muppets-1.jpeg"
    #image_file = "test_images/horses.jpg"
    #image_file = "test_images/dog.jpg"

    # model file is generated by stuff in the keras-YOLOv3-model-set repo, see note near top
    model_file = None
    #model_file = "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.int8.tflite-2023-02-15.tflite"
    #model_file = "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.int8_zero_restricted.tflite"
    #model_file = "/home/volker/Documents/806-yolo_decode/tiny_yolo_v3size320.ikva_zero_restricted.tflite"

    raw_output_file = None
    #raw_output_file = "/home/volker/Documents/806-yolo_decode/raw_briey_zero_int8"
    raw_output_file = "/home/volker/Documents/806-yolo_decode/raw_ikva_zero_int8"

    # decodes to something similar, but still different...
    blocks_file = None
    #blocks_file = "/home/volker/Documents/806-yolo_decode/blocks_int8_briey-2023-02-15-good"
    #blocks_file = "/home/volker/Documents/806-yolo_decode/blocks_int8_ref-2023-01-31"
    #blocks_file = "/home/volker/Documents/806-yolo_decode/blocks_briey_zero_int8"


    if show_boxes and (len(show_boxes) > 0):
        # just show the boxes
        print("\n>>>>>>> JUST SHOWING BOXES <<<<<<<<<<<<\n")
        cmp_boxes(show_boxes, ref_boxes)
        draw_boxes(image_file, show_boxes, scale=True, ref_boxes=ref_boxes)

    else:
        if DUMP_RAW:        fname = "raw_python"
        elif DUMP_BLOCKS:   fname = "blocks_python"
        elif DUMP_OBJNESS:  fname = "objness_python"
        elif DUMP_P_CLASS:  fname = "p_class_python"
        elif DUMP_COORDS:   fname = "coords_python"
        else:               assert not DUMP_SOMETHING

        if DUMP_SOMETHING:
            assert fname != ""
            outf = open(fname, "w")
            assert outf

        raw_outputs = None

        anchors = {
            'StatefulPartitionedCall:0' : np.array([(10,14), (23,27), (37,58)]),
            'StatefulPartitionedCall:1' : np.array([(81,82), (135,169), (344,319)])
        }

        # default values (if model is run, these are read from the model!)
        DEQUANT_10_Q = 0.1762162148952484
        DEQUANT_10_OFF = 0.0
        DEQUANT_20_Q = 0.17356666922569275
        DEQUANT_20_OFF = 0.0

        if model_file:
            img_array = pre_process_image(image_file)
            #dump_array("network_input.h", "int8_t", img_array)

            raw_outputs = get_interpreter_output(model_file, img_array, anchors)

        elif raw_output_file:
            print(f"\n>>>> NOT RUNNING INFERENCE - de-coding net output from blocks {blocks_file}\n")
            exec(open(raw_output_file).read())

            raw_outputs = [
                InterpreterOutput(
                    name = 'StatefulPartitionedCall:1',
                    input_shape = [NET_IN_SZ, NET_IN_SZ],
                    output_shape = [10, 10],
                    anchors = np.array([(81,82), (135,169), (344,319)]),
                    data = raw_10x10,
                    quant_off = DEQUANT_10_OFF,
                    quant_scale = DEQUANT_10_Q
                ),

                InterpreterOutput(
                    name = 'StatefulPartitionedCall:0',
                    input_shape = [NET_IN_SZ, NET_IN_SZ],
                    output_shape = [20, 20],
                    anchors = np.array([(10,14), (23,27), (37,58)]),
                    data = raw_20x20,
                    quant_off = DEQUANT_20_OFF,
                    quant_scale = DEQUANT_20_Q
                )
            ]


        if raw_outputs:
            make_blocks(raw_outputs)
            blocks = raw_outputs

        elif blocks_file:
            print(f"\n>>>> NOT RUNNING INFERENCE - de-coding net output from blocks {blocks_file}\n")
            exec(open(blocks_file).read())
            blocks = [
                InterpreterOutput(
                    name = 'StatefulPartitionedCall:1',
                    input_shape = [NET_IN_SZ, NET_IN_SZ],
                    output_shape = [10, 10],
                    anchors = np.array([(81,82), (135,169), (344,319)]),
                    data = blocks_10x10,
                    quant_off = DEQUANT_10_OFF,
                    quant_scale = DEQUANT_10_Q
                ),

                InterpreterOutput(
                    name = 'StatefulPartitionedCall:0',
                    input_shape = [NET_IN_SZ, NET_IN_SZ],
                    output_shape = [20, 20],
                    anchors = np.array([(10,14), (23,27), (37,58)]),
                    data = blocks_20x20,
                    quant_off = DEQUANT_20_OFF,
                    quant_scale = DEQUANT_20_Q
                )
            ]

        else:
            assert False, "need blocks file or raw outputs"

        bboxes = decode_yolo(blocks)
        bboxes = bboxes[:USE_MAX_BOXES]
        summarise_boxes(bboxes)

        for b in bboxes:
            print(b)

        if ref_boxes and (len(ref_boxes) > 0):
            cmp_boxes(bboxes, ref_boxes)

        if len(bboxes) > 0:
            draw_boxes(image_file, bboxes, scale=True, ref_boxes=ref_boxes)

        if DUMP_SOMETHING:
            assert outf
            outf.close()