import time
from absl import app, flags, logging
from absl.flags import FLAGS

import numpy as np
import cv2
import tensorflow as tf
import sys, path
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

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

def make_io_panels(zctxt):
    input_panel = zctxt.panel('Input', width=0.25, height=0.3, screen_x=0.0, screen_y=0.7, navigation='2d').set_camera(position=(
        1, 0.75, 30), aim=(1, 0.75, 0)).set_background_color(rgb=(0.025, 0.02, 0.045)).border(3).set_border_alpha(0.05)
    output_panel = zctxt.panel('Output', width=0.25, height=0.3, screen_x=0.0, screen_y=0.0, navigation='2d').set_camera(position=(
        1, 0.75, 30), aim=(1, 0.75, 0)).set_background_color(rgb=(0.025, 0.02, 0.045)).border(3).set_border_alpha(0.05)
    zctxt.text("Input").font_size(0.1).position(y=-.45).send_to(input_panel)
    zctxt.text("Output").font_size(.1).position(y=-.45).send_to(output_panel)
    return input_panel, output_panel

def api(video_file=""):
    runner(video_file)

def main(_argv):
    runner()

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
    input_panel, output_panel = make_io_panels(ctxt)

    counter = 0

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

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
            to_fit = 0.15 / image_np.shape[2]
            zinput = ctxt.image().data(image_np).scale(to_fit, to_fit).send_to(input_panel)
            zmodel = ctxt.model().keras(yolo).inputs(img_in.numpy())
            ctxt.snapshot()
        else:
            zinput.data(image_np).update()
            zmodel.inputs(img_in.numpy()).update()

        t1 = time.time()
        #boxes, scores, classes, nums = yolo.predict(img_in)
        bbox, confidence, class_probs, scores = yolo(img_in)
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

        if counter == 0:
            to_fit = 0.15 / out_img.shape[2]
            zoutput = ctxt.image().data(out_img).scale(to_fit, to_fit).send_to(output_panel).update()
        else:
            zoutput.data(out_img).update();

        counter += 1
        if cv2.waitKey(1) == ord('q'):
            break

    ctxt.disconnect()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
