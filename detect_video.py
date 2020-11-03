import time
from absl import app, flags, logging
from absl.flags import FLAGS

import numpy as np
import cv2
import tensorflow as tf
import sys, os
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '/Users/jon/z10/package')))
import zetane

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
image_table = {'person':0, 'bicycle':1, 'car':2, 'motorbike':3, 'aeroplane':4, 'bus':5, 'train':6,
               'truck':7, 'boat':8, 'traffic light':9, 'fire hydrant':10, 'stop sign':11, 'parking meter':12,
               'bench':13, 'bird':14, 'cat':15, 'dog':16, 'horse':17, 'sheep':18,
               'cow':19, 'elephant':20, 'bear':21, 'zebra':22, 'giraffe':23, 'backpack':24,
               'umbrella':25, 'handbag':26, 'tie':27, 'suitcase':28, 'frisbee':29, 'skis':30, 
               'snowboard':31, 'sports ball':32, 'kite':33, 'baseball bat':34, 'baseball glove':35, 'skateboard':36,
               'surfboard':37, 'tennis racket':38, 'bottle':39, 'wine glass':40, 'cup':41, 'fork':42,
               'knife':43, 'spoon':44, 'bowl':45, 'banana':46, 'apple':47, 'sandwich':48, 
               'orange':49, 'broccoli':50, 'carrot':51, 'hot dog':52, 'pizza':53, 'donut':54,
               'cake':55, 'chair':56, 'sofa':57, 'pottedplant':58, 'bed':59, 'diningtable':60,
               'toilet':61, 'tvmonitor':62, 'laptop':63, 'mouse':64, 'remote':65, 'keyboard':66,
               'cell phone':67, 'microwave':68, 'oven':69, 'toaster':70, 'sink':71, 'refrigerator':72,
               'book':73, 'clock':74, 'vase':75, 'scissors':76, 'teddy bear':77, 'hair drier':78, 'toothbrush':79}

def image_map(classes,image_table, N=1):
    image_table = image_table
    image_keys = list(image_table.keys())
    images = []
    for i in range(N):
        images.append(image_keys[int(classes[i])])

    return images

def api(video_file=""):
    runner(video_file)

def make_io_panels(zctxt):
    input_panel = zctxt.panel('Input', width=1.0, height=0.3, screen_x=0.0, screen_y=0.7, navigation='2d').set_camera(position=(1, 0.75, 30), aim=(1, 0.75, 0)).set_background_color(rgb=(0.025, 0.02, 0.045)).border(3).set_border_alpha(0.05)
    return input_panel

def main(_argv):
    runner()

def build_class_list(scores, classes):
    scores_list = scores.numpy().tolist()
    class_list = classes.numpy().tolist()
    print(scores_list)
    print(class_list)
    class_list_map = []
    target_output_list = []

    for i in range(len(class_list[0])):
        if (scores_list[0][i] != 0.0): 
            class_list_map.append(class_list[0][i])
            target_output = image_map(class_list_map, image_table)
            target_output_list.append(target_output)
            class_list_map = []

            print(target_output_list)
            print("This is final target list of classes", target_output_list)

    return target_output_list, scores_list

def predict(img_in, yolo, times, img, class_names):
    t1 = time.time()
    bbox, confidence, class_probs, scores = yolo.predict(img_in)
    boxes, scores, classes, nums = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )
    t2 = time.time()
    times.append(t2-t1)
    times = times[-20:]

    out_img = draw_outputs(img/255.0, (boxes, scores, classes, nums), class_names)
    out_img = cv2.putText(out_img, "Time: {:.2f}ms".format(sum(times)/len(times)*1000), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    out_img = cv2.resize(out_img, (416, 416))

    return scores, classes, out_img

def runner(video_file=""):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    if len(video_file) == 0:
        try:
            video_file = int(FLAGS.video)
        except:
            video_file = FLAGS.video

    vid = cv2.VideoCapture(video_file)

    out = None

    ctxt = zetane.Context(remote=True)
    #input_panel = make_io_panels(ctxt)
    snapstream = ctxt.stream()

    counter = 0

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    z_txt_classes = ctxt.text()
    z_txt_scores = ctxt.text()

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        image_np = np.transpose(img_in.numpy(), (1, 2, 3, 0))

        if counter == 0:
            to_fit = 0.04 / image_np.shape[2]
            zinput = ctxt.image().data(image_np).position(-0.3,-0.5, 0).scale(to_fit,to_fit*4)
            #zmodel = ctxt.model().keras(yolo).inputs(img_in.numpy())
            zoutput = ctxt.image().position(0.5,-0.5, 0)
        else:
            zinput.data(image_np)
            #zmodel.inputs(img_in.numpy())

        if counter % 10 == 0:
            scores, classes, out_img = predict(img_in, yolo, times, img, class_names)
            target_output_list, scores_list = build_class_list(scores, classes)

            #z_txt_classes.text("CLASSES" +'\n'+ str(target_output_list[0][0])+'\n'+ str(target_output_list[1][0])+'\n').position(1.5, 1.7, 0).scale(0.2,0.2,0.2)
            #z_txt_scores.text("SCORES" +'\n'+str(round(scores_list[0][0], 3))+'\n'+ str(round(scores_list[0][1], 3))+'\n').position(2.0, 1.7, 0).scale(0.2,0.2,0.2)
            to_fit = 0.04 / out_img.shape[2]
            zoutput.data(out_img).position(0.5,-0.5, 0).scale(to_fit, to_fit*4)

        ctxt.snapshot(stream=snapstream)
        counter += 1

        if cv2.waitKey(1) == ord('q'):
            snapstream.serialize()
            break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
