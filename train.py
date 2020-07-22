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


def train():
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
    train_dataset, test_dataset, num_train_samples, num_test_samples, classes = get_voc_dataset('train', args.batch_size)
    logger.info("Number of train samples: {},  Number of test samples: {},  Number of classes = {}".format(num_train_samples, num_test_samples, len(classes)-1))
    
    # create model
    logger.info("Creating Model")
    model, hyper_params = utils_model.get_model(args.arch, len(classes), args.weights)
    logger.info("Model Created")    
    
    
    # loss function and optimizer
    loss_fn = CustomLoss(3, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # We calculate prior boxes for one time and use it for all operations because all images are the same sizes
    prior_boxes = utils_bbox.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
    
    ######### 
    model.load_weights("weights.h5")
    
    
    
    
    
    # create meters to measure mean
    mean_loss = Mean()
    mean_conf_loss = Mean()
    mean_loc_loss = Mean()
    mean_l2_loss = Mean()
    
    logger.info('Starting Training')
    for epoch in range(1, args.epochs+1):
        logger.info("Starting epoch {}".format(epoch))
        wandb.log({'epoch': epoch})
        
        ########## TRAINING ##########
        mean_loss.reset_states()
        mean_conf_loss.reset_states()
        mean_loc_loss.reset_states()
        mean_l2_loss.reset_states()
        start = time()
        last = start

        for step, image_data in enumerate(train_dataset, 1):
            imgs, gt_boxes, gt_labels = image_data
            gt_locs, gt_confs = utils_train.calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params)
            
            loss, loc_loss, conf_loss, l2_loss = train_step(model, imgs, gt_locs, gt_confs, loss_fn, optimizer)
            mean_loss.update_state(loss)
            mean_conf_loss.update_state(conf_loss)
            mean_loc_loss.update_state(loc_loss)
            mean_l2_loss.update_state(l2_loss)

            if (step % args.log_interval) == 0:
                wandb.log({'loss':mean_loss.result(), 'conf_loss':mean_conf_loss.result(), 'loc_loss':mean_loc_loss.result(), 'l2_loss':mean_l2_loss.result()})
                speed = (args.batch_size*args.log_interval)/(time()-last)
                last = time()
                logger.info('Epoch[{}/{}] {:>3}/{:3}  Speed: {:5.2f}img/s   |   Loss: {:6.3f}   Conf: {:6.3f}   Loc: {:6.3f}   L2: {:6.3f}'
                      .format(epoch, args.epochs, step, math.ceil(num_train_samples/args.batch_size), speed, mean_loss.result(), mean_conf_loss.result(), mean_loc_loss.result(), mean_l2_loss.result()))
            
        ########## VALIDATION ##########
        if epoch % args.val_interval == 0:
            start = time()
            mean_loss.reset_states()
            mean_conf_loss.reset_states()
            mean_loc_loss.reset_states()
            mean_l2_loss.reset_states()
            
            for step, image_data in enumerate(test_dataset, 1):
                imgs, gt_boxes, gt_labels = image_data
                gt_locs, gt_confs = utils_train.calculate_actual_outputs(prior_boxes, gt_boxes, gt_labels, hyper_params)

                loss, loc_loss, conf_loss, l2_loss = val_step(model, imgs, gt_locs, gt_confs, loss_fn)
                mean_loss.update_state(loss)
                mean_conf_loss.update_state(conf_loss)
                mean_loc_loss.update_state(loc_loss)
                mean_l2_loss.update_state(l2_loss)

            val_time = time() - start
            logger.info('### VAL ### Epoch[{}/{}]  val_time: {:5.2f}s |  Loss: {:6.3f} Conf: {:6.3f} Loc: {:6.3f} L2: {:7.6f}'
                  .format(epoch, args.epochs, val_time, mean_loss.result(), mean_conf_loss.result(), mean_loc_loss.result(), mean_l2_loss.result()))
    
    
    # calculate final mAP
    ssd_decoder_model = get_decoder_model(model, prior_boxes, hyper_params)
    mAP = test_mAP(ssd_decoder_model, test_dataset, num_test_samples, classes, args.batch_size)
    logger.info("mAP: {:5.2f}%".format(float(mAP*100)))
    wandb.run.summary.update({"mAP": 100*mAP})
    
    # save last only for now
    model.save_weights(os.path.join(args.save_dir, 'ssd_epoch_{}.h5'.format(epoch)))
    
        
    
@tf.function
def train_step(model, imgs, gt_locs, gt_confs, loss_fn, optimizer):
    with tf.GradientTape() as tape:

        # forward pass
        locs, confs = model(imgs) 

        # main loss
        loc_loss, conf_loss = loss_fn.loc_loss_fn(locs, gt_locs), loss_fn.conf_loss_fn(confs, gt_confs)
        loss = loc_loss + conf_loss

        # Add any extra losses created during the forward pass.
        l2_loss = sum(model.losses)
        loss += l2_loss

    # get gradients from loss & tape
    grads = tape.gradient(loss, model.trainable_weights)

    # apply gradients
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    return loss, loc_loss, conf_loss, l2_loss


@tf.function
def val_step(model, imgs, gt_locs, gt_confs, loss_fn):

    # forward pass
    locs, confs = model(imgs) 

    # main loss
    loc_loss, conf_loss = loss_fn.loc_loss_fn(locs, gt_locs), loss_fn.conf_loss_fn(confs, gt_confs)
    loss = loc_loss + conf_loss

    # Add any extra losses created during the forward pass.
    l2_loss = sum(model.losses)
    loss += l2_loss
    
    return loss, loc_loss, conf_loss, l2_loss


def test_mAP(ssd_decoder_model, test_dataset, num_test_samples, classes, batch_size):
    logger.info("Calculating mAP on VOC2007-test")
    step_size = utils_train.get_step_size(num_test_samples, batch_size)
    
    pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict(test_dataset, steps=step_size, verbose=1)
    mAP = utils_eval.evaluate_predictions(test_dataset, pred_bboxes, pred_labels, pred_scores, classes, batch_size)
    
    return mAP


if __name__ == '__main__':
    train()