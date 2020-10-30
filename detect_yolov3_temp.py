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
    input_panel = zctxt.panel('', width=1.0, height=0.3, screen_x=0.0, screen_y=0.7, navigation='2d').set_camera(position=(
        1, 0.75, 30), aim=(1, 0.75, 0)).set_background_color(rgb=(0.025, 0.02, 0.045)).border(3).set_border_alpha(0.05).update()
    #output_panel = zctxt.panel('Output', width=0.25, height=0.3, screen_x=0.0, screen_y=0.0, navigation='2d').set_camera(position=(
    #    1, 0.75, 30), aim=(1, 0.75, 0)).set_background_color(rgb=(0.025, 0.02, 0.045)).border(3).set_border_alpha(0.05).update()
    #zctxt.text("ABC").scale(10, 10, 10).send_to(input_panel)
    #zctxt.text("ABC").font_size(0.1).position(y=-.45).send_to(input_panel)
    #zctxt.text("Output").font_size(.1).position(y=-.45).scale(0.1,0.1,0.1).send_to(output_panel).update()
    #return input_panel, output_panel
    return input_panel

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
    print(class_names)
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

    #input_panel, output_panel = make_io_panels(ctxt)
    input_panel = make_io_panels(ctxt)
    image_np = np.transpose(img.numpy(), (1, 2, 3, 0))
    print(image_np.shape)
    to_fit = 0.04 / image_np.shape[2]
    print(image_np.shape[2])
    print("This is the print of to_fit", to_fit)
    #fixed_image_height = 120
    #img_height = image_np.shape[2]
    #img_scl = fixed_image_height / img_height
    zinput = ctxt.image().data(image_np).position(-0.3,-0.5, 0).scale(to_fit,to_fit*4).send_to(input_panel).update()
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

    print("This is class prob",class_probs)
    print("This is class prob shape", class_probs.shape)
    print("This is confidence", confidence)
    print("This is the scores", scores)
    scores_list = scores.numpy().tolist()
    print(type(scores_list))
    print("This is the numpy list", str(scores_list[0][1]))
    print("This is the classes", classes)
    class_list = classes.numpy().tolist()
    print(len(class_list[0]))
    for i in range(len(class_list[0])):
        print(int(class_list[0][i]))

    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(nums[0]):
        logging.info('This is the info \t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))
    
    def image_map(classes,image_table, N=1):
        """Take the most probable labels (output of postprocess).
        Args:
            classes (list): names of the classes used for training the model
            image_table (dictionary): dictionary to map the id to the names of the classes 
            N (int):  top N labels that fit the picture
            
        Returns:
            images (list):top N labels that fit the picture.
        """
        
        image_table = image_table
        
        image_keys = list(image_table.keys())
        images = []
        for i in range(N):
            images.append(image_keys[int(classes[i])])
            
        return images
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
    
    
    
    class_list_map = []
    target_output_list = []
    
            
    for i in range(len(class_list[0])):
        
        if (scores_list[0][i] != 0.0): 
            class_list_map.append(class_list[0][i])
            #target_output = image_map(list(map(int, str(int(class_list[0][i])))), image_table)
            target_output = image_map(class_list_map, image_table)
            target_output_list.append(target_output)
            class_list_map = []
            print(target_output_list)
    
    

    #out_img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    #out_img = draw_outputs(img_raw.numpy()/225.0, (boxes, scores, classes, nums), class_names)
    out_img = draw_outputs(img_raw.numpy()/255.0, (boxes, scores, classes, nums), class_names)
    out_img = cv2.resize(out_img, (416, 416))
    print(out_img.shape)
    new_fit = 0.04 / out_img.shape[2]
    
    
    print(out_img.shape[2])
    print("The output fit size", to_fit)
    zoutput = ctxt.image().data(out_img).position(0.5,-0.5, 0).scale(new_fit, new_fit*4).send_to(input_panel).update()
    z_txt_classes = ctxt.text().send_to(input_panel)
    z_txt_scores = ctxt.text().send_to(input_panel)
    
    z_txt_classes.text("CLASSES" +'\n'+ str(target_output_list[0][0])+'\n'+ str(target_output_list[1][0])+'\n'+ str(target_output_list[2][0])+'\n'+ str(target_output_list[3][0])+'\n'+ str(target_output_list[4][0])).position(1.5, 1.7, 0).scale(0.2,0.2,0.2).update()
    z_txt_scores.text("SCORES" +'\n'+str(round(scores_list[0][0], 3))+'\n'+ str(round(scores_list[0][1], 3))+'\n'+ str(round(scores_list[0][2], 3))+'\n'+ str(round(scores_list[0][3], 3))+'\n'+ str(round(scores_list[0][4], 3))).position(2.0, 1.7, 0).scale(0.2,0.2,0.2).update()
    
    
    #cv2.imwrite(FLAGS.output, img)
    #logging.info('output saved to: {}'.format(FLAGS.output))

    ctxt.disconnect()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass