import argparse
import os
import logging
logging.basicConfig(level=logging.ERROR, format='%(message)s') # ERROR to suppress deprecated warning for distorted boundig box message...
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # limit tensorflow output to terminal

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow_datasets as tfds
from datasets import voc
import datasets.augmentation
from utils import utils_bbox, utils_data, utils_io, utils_train, utils_model, utils_eval
from datasets.voc import get_voc_dataset
from tensorflow.keras.metrics import Mean
from loss.ssd_loss import CustomLoss
from time import time
import math
from models.decoder import get_decoder_model
import wandb

def parse():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='SSD_Mobilenet2')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--pretrained-type', default='base')
    parser.add_argument('--data-dir', default='/home/dataset/tensorflow_datasets')
    parser.add_argument('--initial-lr', default=1e-3, type=float)
    parser.add_argument('--weights',default=None)
    

    parser.add_argument('--neg-ratio', default=3, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--log-interval', default=10, type=int)
    parser.add_argument('--val-interval', default=1, type=int)
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save-dir', default='/home/work2/weights')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    
    return args


def test():
    ''' main pipeline '''
    
    # parse arguments
    args = parse()
    
    # limit visible gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # wandb
    if args.dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
    if args.name:
        wandb.init(project="tf2-object-detection", config=args, dir="/home/work2/wandb", name=args.name)
    else:
        wandb.init(project="tf2-object-detection", config=args, dir="/home/work2/wandb")

    # create dataset
    logger.info("Creating Datasets")
    test_dataset, num_test_samples, classes = get_voc_dataset('test', args.batch_size)
    logger.info("Number of test samples: {},  Number of classes = {}".format(num_test_samples, len(classes)-1))
  
    # create model
    logger.info("Creating Model")
    model, hyper_params = utils_model.get_model(args.arch, args.pretrained_type, len(classes), args.neg_ratio, args.weights, args.initial_lr)
    logger.info("Model Created")
    
    # loss function and optimizer
    loss_fn = CustomLoss(3, 1)
    
    # We calculate prior boxes for one time and use it for all operations because all images are the same sizes
    prior_boxes = utils_bbox.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"]) 

    
    # calculate final mAP
    ssd_decoder_model = get_decoder_model(model, prior_boxes, hyper_params)
    
    logger.info("Calculating mAP on VOC2007-test")
    step_size = utils_train.get_step_size(num_test_samples, args.batch_size)
    logger.info("step_size={}".format(step_size))
    
    pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(test_dataset, steps=step_size, verbose=1)
    logger.info(pred_bboxes.shape)
    logger.info("pred_bboxes example:")
    logger.info(pred_bboxes)
    logger.info(pred_labels)
    logger.info(pred_scores)
    mAP = utils_eval.evaluate_predictions(test_dataset, pred_bboxes, pred_labels, pred_scores, classes, args.batch_size)
    logger.info(mAP)
    
    logger.info("mAP: {:5.2f}%".format(float(mAP*100)))
    wandb.run.summary.update({"mAP": 100*mAP})
    
    # save last only for now
    model.save_weights(os.path.join(args.save_dir, 'ssd_epoch_{}.h5'.format(epoch)))
    



if __name__ == '__main__':
    test()