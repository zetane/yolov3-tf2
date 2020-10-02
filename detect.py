import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import zetane

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def make_io_panels(zctxt):
    input_panel = zctxt.panel('Input', width=0.25, height=0.3, screen_x=0.0, screen_y=0.7, navigation='2d').set_camera(position=(
        1, 0.75, 30), aim=(1, 0.75, 0)).set_background_color(rgb=(0.025, 0.02, 0.045)).border(3).set_border_alpha(0.05).update()
    output_panel = zctxt.panel('Output', width=0.25, height=0.3, screen_x=0.0, screen_y=0.0, navigation='2d').set_camera(position=(
        1, 0.75, 30), aim=(1, 0.75, 0)).set_background_color(rgb=(0.025, 0.02, 0.045)).border(3).set_border_alpha(0.05).update()
    zctxt.text("Input").font_size(0.1).position(y=-.45).send_to(input_panel).update()
    zctxt.text("Output").font_size(.1).position(y=-.45).send_to(output_panel).update()
    return input_panel, output_panel

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, FLAGS.size)

    ctxt = zetane.Context()
    ctxt.clear_universe()

    input_panel, output_panel = make_io_panels(ctxt)
    image_np = np.transpose(img.numpy(), (1, 2, 3, 0))
    to_fit = 0.15 / image_np.shape[2]
    zinput = ctxt.image().data(image_np).scale(to_fit, to_fit).send_to(input_panel).update()
    zmodel = ctxt.model().keras(yolo).inputs(img.numpy()).update()

    t1 = time.time()
    #boxes, scores, classes, nums = yolo(img)
    bbox, confidence, class_probs, scores = yolo(img)
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
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))

    #out_img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    out_img = draw_outputs(img_raw.numpy()/225.0, (boxes, scores, classes, nums), class_names)

    to_fit = 0.15 / out_img.shape[2]
    zoutput = ctxt.image().data(out_img).scale(to_fit, to_fit).send_to(output_panel).update()
    #cv2.imwrite(FLAGS.output, img)
    #logging.info('output saved to: {}'.format(FLAGS.output))

    ctxt.disconnect()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
