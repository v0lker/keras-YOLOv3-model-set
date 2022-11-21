#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a YOLOv3/YOLOv2 style detection model on test images.
"""

# import colorsys
import os, sys, argparse
# import cv2
import time
# from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Input, Lambda
# # from tensorflow_model_optimization.sparsity import keras as sparsity
# from tensorflow import keras as sparsity
from PIL import Image

# # from yolo5.model import get_yolo5_model, get_yolo5_inference_model
# # from yolo5.postprocess_np import yolo5_postprocess_np
from yolo3.model import get_yolo3_model, get_yolo3_inference_model
from yolo3.postprocess_np import yolo3_postprocess_np
# # from yolo2.model import get_yolo2_model, get_yolo2_inference_model
# # from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes
# #from tensorflow.keras.utils import multi_gpu_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#tf.enable_eager_execution()

default_config = {
        "model_type": 'tiny_yolo3_darknet',
        "weights_path": os.path.join('weights', 'yolov3-tiny.h5'),
        "pruning_model": False,
        "anchors_path": os.path.join('configs', 'tiny_yolo3_anchors.txt'),
        "classes_path": os.path.join('configs', 'coco_classes.txt'),
        "score" : 0.1,
        "iou" : 0.4,
        "model_input_shape" : (416, 416),
        "elim_grid_sense": False,
        #"gpu_num" : 1,
    }


class YOLO_np(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO_np, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(len(self.class_names))
        K.set_learning_phase(0)
        self.yolo_model = self._generate_model()

    def _generate_model(self) -> "tf.keras.engine.functional.Functional":
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        try:
            if self.model_type.startswith('scaled_yolo4_') or self.model_type.startswith('yolo5_'):
                # Scaled-YOLOv4 & YOLOv5 entrance
                yolo_model, _ = get_yolo5_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_input_shape + (3,), model_pruning=self.pruning_model)
            elif self.model_type.startswith('yolo3_') or self.model_type.startswith('yolo4_') or \
                 self.model_type.startswith('tiny_yolo3_') or self.model_type.startswith('tiny_yolo4_'):
                # YOLOv3 & v4 entrance
                yolo_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_input_shape + (3,), model_pruning=self.pruning_model)
            elif self.model_type.startswith('yolo2_') or self.model_type.startswith('tiny_yolo2_'):
                # YOLOv2 entrance
                yolo_model, _ = get_yolo2_model(self.model_type, num_anchors, num_classes, input_shape=self.model_input_shape + (3,), model_pruning=self.pruning_model)
            else:
                raise ValueError('Unsupported model type')

            yolo_model.load_weights(weights_path) # make sure model, anchors and classes match
            if self.pruning_model:
                yolo_model = sparsity.strip_pruning(yolo_model)
            yolo_model.summary()
        except Exception as e:
            print(repr(e))
            assert yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'
        print('{} model, anchors, and classes loaded.'.format(weights_path))
        #if self.gpu_num>=2:
            #yolo_model = multi_gpu_model(yolo_model, gpus=self.gpu_num)

        return yolo_model


    def detect_image(self, image):
        if self.model_input_shape != (None, None):
            assert self.model_input_shape[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_input_shape[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_input_shape)
        #origin image shape, in (height, width) format
        image_shape = image.size[::-1]

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        for box, score in zip(out_boxes, out_scores):
            print(f"{box} ({score})")
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores


    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        # raise ValueError(self.yolo_model.input_shape)
        # raise ValueError(self.yolo_model.output_shape)
        outputs = self.yolo_model.predict(image_data)
        o1, o2 = outputs
        # self.yolo_model.save("a.h5")
        # raise ValueError("siema", o1.shape, o2.shape)
        out_boxes, out_classes, out_scores = yolo3_postprocess_np(outputs, image_shape, self.anchors, len(self.class_names), self.model_input_shape, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)

        return out_boxes, out_classes, out_scores


    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)



class YOLO(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(len(self.class_names))
        K.set_learning_phase(0)
        self.inference_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        inference_model = get_yolo3_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_input_shape + (3,), confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)

        inference_model.summary()
        return inference_model

    def predict(self, image_data, image_shape):
        out_boxes, out_scores, out_classes = self.inference_model.predict([image_data, image_shape])

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image):
        if self.model_input_shape != (None, None):
            assert self.model_input_shape[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_input_shape[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_input_shape)

        # prepare origin image shape, (height, width) format
        image_shape = np.array([image.size[1], image.size[0]])
        image_shape = np.expand_dims(image_shape, 0)

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        end = time.time()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores

    def dump_model_file(self, output_model_file):
        self.inference_model.save(output_model_file)

    def dump_saved_model(self, saved_model_path):
        model = self.inference_model
        os.makedirs(saved_model_path, exist_ok=True)

        tf.keras.experimental.export_saved_model(model, saved_model_path)
        print('export inference model to %s' % str(saved_model_path))




from pathlib import Path
def detect_img(yolo, img: Path):
    image = Image.open(img).convert('RGB')
    r_image, _, _, _ = yolo.detect_image(image)
    r_image: Image
    r_image.save("a.jpg")
    # r_image.show()


def main():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo or dump out YOLO h5 model')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_type', type=str,
        help='YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobilenet/yolo3_darknet/..., default ' + YOLO.get_defaults("model_type")
    )

    parser.add_argument(
        '--weights_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("weights_path")
    )

    parser.add_argument(
        '--pruning_model', default=False, action="store_true",
        help='Whether to be a pruning model/weights file, default ' + str(YOLO.get_defaults("pruning_model"))
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--model_input_shape', type=str,
        help='model image input shape as <height>x<width>, default ' +
        str(YOLO.get_defaults("model_input_shape")[0])+'x'+str(YOLO.get_defaults("model_input_shape")[1]),
        default=str(YOLO.get_defaults("model_input_shape")[0])+'x'+str(YOLO.get_defaults("model_input_shape")[1])
    )

    parser.add_argument(
        '--elim_grid_sense', default=False, action="store_true",
        help = "Eliminate grid sensitivity, default " + str(YOLO.get_defaults("elim_grid_sense"))
    )

    #parser.add_argument(
        #'--gpu_num', type=int,
        #help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    #)
    from pathlib import Path
    parser.add_argument(
        '--image', required=True, type=Path)
    
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )
    '''
    Command line positional arguments -- for model dump
    '''
    parser.add_argument(
        '--dump_model', default=False, action="store_true",
        help='Dump out training model to inference model'
    )

    parser.add_argument(
        '--output_model_file', type=str,
        help='output inference model file'
    )

    args = parser.parse_args()
    # param parse
    if args.model_input_shape:
        height, width = args.model_input_shape.split('x')
        args.model_input_shape = (int(height), int(width))
        assert (args.model_input_shape[0]%32 == 0 and args.model_input_shape[1]%32 == 0), 'model_input_shape should be multiples of 32'

    # get wrapped inference object
    yolo = YOLO_np(**vars(args))
    detect_img(yolo, args.image)

if __name__ == '__main__':
    main()
