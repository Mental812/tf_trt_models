import sys
import time

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import logging

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge ,CvBridgeError
from sub_image import CImageSubscriber

from utils.od_utils import  load_trt_pb, detect
from utils.visualization import BBoxVisualization

DEFAULT_MODEL = 'ssd_mobilenet_v1_coco'
DEFAULT_LABELMAP = 'third_party/models/research/object_detection/' \
                   'data/mscoco_label_map.pbtxt'
WINDOW_NAME = 'CameraTFTRTDemo'
BBOX_COLOR = (0, 255, 0)  # green


def read_label_map(path_to_labels):
    """Read from the label map file and return a class dictionary which
    maps class id (int) to the corresponding display name (string).

    Reference:
    https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
    """
    from object_detection.utils import label_map_util

    category_index = label_map_util.create_category_index_from_labelmap(
        path_to_labels)
    cls_dict = {int(x['id']): x['name'] for x in category_index.values()}
    num_classes = max(c for c in cls_dict.keys()) + 1
    # add missing classes as, say,'CLS12' if any
    return {i: cls_dict.get(i, 'CLS{}'.format(i)) for i in range(num_classes)}

def open_display_window(width, height):
    """Open the cv2 window for displaying images with bounding boxeses."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera TFTRT Object Detection Demo '
                                    'for Jetson TX2/TX1')

def draw_help_and_fps(img, fps):
    """Draw help message and fps number at top-left corner of the image."""
    help_text = "'Esc' to Quit, 'H' for FPS & Help, 'F' for Fullscreen"
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA

    fps_text = 'FPS: {:.1f}'.format(fps)
    cv2.putText(img, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    cv2.putText(img, fps_text, (11, 50), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 50), font, 1.0, (240, 240, 240), 1, line)
    return img

def set_full_screen(full_scrn):
    """Set display window to full screen or not."""
    prop = cv2.WINDOW_FULLSCREEN if full_scrn else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

def loop_and_detect(cam,tf_sess, conf_th, vis, od_type):
    """Loop, grab images from camera, and do object detection.

    # Arguments
      cam: the camera object (video source).
      tf_sess: TensorFlow/TensorRT session to run SSD object detection.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    show_fps = True
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the display window.
            # If yes, terminate the while loop.
            break
        
        img = cam.getImage()
        if img is not None:
            box, conf, cls = detect(img, tf_sess, conf_th, od_type=od_type)
            img = vis.draw_bboxes(img, box, conf, cls)
            if show_fps:
                img = draw_help_and_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.9 + curr_fps*0.1)
            tic = toc

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # Toggle help/fps
            show_fps = not show_fps
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_full_screen(full_scrn)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logging.getLogger('tensorflow').propagate = False


    logger.info('reading label map')
    cls_dict = read_label_map(DEFAULT_LABELMAP)

    pb_path = './data/{}_trt.pb'.format(DEFAULT_MODEL)
    log_path = './logs/{}_trt'.format(DEFAULT_MODEL)
    
    logger.info('opening ros camera device/file')

    cam = CImageSubscriber("/camera/color/image_raw")

    logger.info('loading TRT graph from pb: %s' % pb_path)
    trt_graph = load_trt_pb(pb_path)

    logger.info('starting up TensorFlow session')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config, graph=trt_graph)

    logger.info('warming up the TRT graph with a dummy image')
    od_type = 'faster_rcnn' if 'faster_rcnn' in DEFAULT_MODEL else 'ssd'
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, _, _ = detect(dummy_img, tf_sess, conf_th=.3, od_type=od_type)

    logger.info('starting to loop and detect')
    detection_conf_th =0.3
    vis = BBoxVisualization(cls_dict)
    open_display_window(640, 480)
    loop_and_detect(cam,tf_sess,  detection_conf_th, vis, od_type=od_type)

    logger.info('cleaning up')
    tf_sess.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()