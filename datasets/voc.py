import tensorflow as tf
import tensorflow_datasets as tfds
tf.random.set_seed(123)
from utils import utils_data
from datasets import augmentation


def get_voc_dataset(batch_size):
    ''' return tfds dataset with preprocessing mapped '''
    
    # train = train2007 + val2007 + train2012 + val2012
    trainval_2007, info_2007 = tfds.load('voc/2007', data_dir='/home/dataset/tensorflow_datasets', split='train+validation', with_info=True)
    trainval_2012, info_2012 = tfds.load('voc/2012', data_dir='/home/dataset/tensorflow_datasets',split='train+validation', with_info=True)
    train_dataset = trainval_2007.concatenate(trainval_2012)
    
    # test = test2007
    test_dataset = tfds.load('voc/2007', split='test', shuffle_files=False)
    
    # number of samples in each dataset
    num_train_samples = int(tf.data.experimental.cardinality(train_dataset))
    num_test_samples = int(tf.data.experimental.cardinality(test_dataset))  
    
    # get list of label names & add background label
    labels = info_2007.features["labels"].names
    labels = ["background"] + labels    
    img_size = 300
    
    # set preprocessing for dataset
    train_dataset = train_dataset.map(lambda x : utils_data.preprocess(x, img_size, img_size, augmentation.apply))
    test_dataset = test_dataset.map(lambda x : utils_data.preprocess(x, img_size, img_size))
    
    # shuffling & batching
    data_shapes = utils_data.get_data_shapes()
    padding_values = utils_data.get_padding_values()
    train_dataset = train_dataset.shuffle(batch_size*4).padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
    test_dataset = test_dataset.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
    
    return train_dataset, test_dataset, num_train_samples, num_test_samples, labels
