import os
import sys
import random

import numpy as np
import tensorflow as tf

import xml.etree.ElementTree as ET

from dataset_utils import int64_feature, float_feature, bytes_feature
from imagenet_common import VOC_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'imagenet_annotation/'
DIRECTORY_IMAGES = 'ilsvrc/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 1000


def _process_image(image_name):

    # Read the image file.
    filename = os.path.join(DIRECTORY_IMAGES,image_name.split('_')[0],image_name)
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(DIRECTORY_ANNOTATIONS, image_name.split('_')[0],(image_name.split('.')[0]+'.xml'))
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,
                        difficult, truncated):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(image_name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.
    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(image_name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%04d.tfrecord' % (output_dir, name, idx)


def run(output_dir, name='imagenet_train', shuffling=True):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """

    # Dataset filenames, and shuffling.
    dictionary_names = sorted(os.listdir(DIRECTORY_IMAGES))
    image_names=[]
    for image_name in dictionary_names:
        image_names += os.listdir(os.path.join(DIRECTORY_IMAGES,image_name))
        
    
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(image_names)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(image_names):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            sys.stdout.write('\r>> Converting dataset %d/%d' % (fidx+1, len(image_names)//SAMPLES_PER_FILES))
            sys.stdout.flush()
            while i < len(image_names) and j < SAMPLES_PER_FILES:
                
                image_name = image_names[i]
                label_path = os.path.join(DIRECTORY_ANNOTATIONS, image_name.split('_')[0],(image_name.split('.')[0]+'.xml'))
                
                try:
                    _add_to_tfrecord(image_name, tfrecord_writer)
                except:
                    i += 1
                    continue 
                i += 1
                j += 1
            fidx += 1
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the ImageNet dataset!')
    
def run_test(output_dir, name='imagenet_train', shuffling=True):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """

    # Dataset filenames, and shuffling.
    dictionary_names = sorted(os.listdir(DIRECTORY_IMAGES))
    image_names=[]
    for image_name in dictionary_names:
        image_names += os.listdir(os.path.join(DIRECTORY_IMAGES,image_name))
        
    
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(image_names)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(image_names):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            sys.stdout.write('\r>> Converting dataset %d/%d' % (fidx+1, len(image_names)//SAMPLES_PER_FILES))
            sys.stdout.flush()
            while i < len(image_names) and j < SAMPLES_PER_FILES:
                
                image_name = image_names[i]
                label_path = os.path.join(DIRECTORY_ANNOTATIONS, image_name.split('_')[0],(image_name.split('.')[0]+'.xml'))
                
                try:
                    _add_to_tfrecord(image_name, tfrecord_writer)
                except:
                    i += 1
                    continue 
                i += 1
                j += 1
            fidx += 1
        break

    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished ImageNet dataset converting test!')
    
if __name__ == '__main__':
    run('imagenet_tfrecord/')
