import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

def preprocess(x, final_height, final_width, augmentation_fn=None, evaluate=False):
    """ Image resizing operation handled before batch operations.]
    Gets a single image with it's bounding boxes & labels and preprocesses them.
    
    inputs:
        image_data = tensorflow dataset image_data
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, channels)
        gt_boxes = (gt_box_size, [y1, x1, y2, x2])
        gt_labels = (gt_box_size)
    """
    
    # get image data
    img = x["image"]
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    
    # get ground truth (bounding boxes & labels)
    # increment all labels because we use 0 for the background class
    gt_box = x["objects"]["bbox"]
    gt_label = tf.cast(x["objects"]["label"] + 1, tf.int32)
    
    # exclude difficult objects for evaluation
    # from official document --> "Objects marked as difficult are currently ignored in the evaluation of the challenge."
    if evaluate:
        not_difficult = tf.logical_not(x["objects"]["is_difficult"])
        gt_box = gt_box[not_difficult]
        gt_label = gt_label[not_difficult]
        
    # apply specified augmentation
    if augmentation_fn:
        img, gt_box = augmentation_fn(img, gt_box)
        
    return img, gt_box, gt_label

def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    """Get tensorflow dataset split and info.
    inputs:
        name = name of the dataset, voc/2007, voc/2012, etc.
        split = data split string, should be one of ["train", "validation", "test"]
        data_dir = read/write path for tensorflow datasets

    outputs:
        dataset = tensorflow dataset split
        info = tensorflow dataset info
    """
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info

def get_total_item_size(info, split):
    """Get total item size for given split.
    inputs:
        info = tensorflow dataset info
        split = data split string, should be one of ["train", "validation", "test"]

    outputs:
        total_item_size = number of total items
    """
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples

def get_labels(info):
    """Get label names list.
    inputs:
        info = tensorflow dataset info

    outputs:
        labels = [labels list]
    """
    return info.features["labels"].names

def get_custom_imgs(custom_image_path):
    """Generating a list of images for given path.
    inputs:
        custom_image_path = folder of the custom images
    outputs:
        custom image list = [path1, path2]
    """
    img_paths = []
    for path, dir, filenames in os.walk(custom_image_path):
        for filename in filenames:
            img_paths.append(os.path.join(path, filename))
        break
    return img_paths

def custom_data_generator(img_paths, final_height, final_width):
    """Yielding custom entities as dataset.
    inputs:
        img_paths = custom image paths
        final_height = final image height after resizing
        final_width = final image width after resizing
    outputs:
        img = (final_height, final_width, depth)
        dummy_gt_boxes = (None, None)
        dummy_gt_labels = (None, )
    """
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)

def get_data_types():
    """Generating data types for tensorflow datasets.
    outputs:
        data types = output data types for (images, ground truth boxes, ground truth labels)
    """
    return (tf.float32, tf.float32, tf.int32)

def get_data_shapes():
    """Generating data shapes for tensorflow datasets.
    outputs:
        data shapes = output data shapes for (images, ground truth boxes, ground truth labels)
    """
    return ([None, None, None], [None, None], [None,])

def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        padding values = padding values with dtypes for (images, ground truth boxes, ground truth labels)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))
