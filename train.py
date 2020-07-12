import argparse
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # limit tensorflow output to terminal

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow_datasets as tfds
from datasets import voc
import datasets.augmentation
from ssd_loss import CustomLoss
from utils import utils_bbox, utils_data, utils_io, utils_train
from datasets.voc import get_voc_dataset


def parse():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='ssd_vgg16')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--pretrained-type', default='base')
    parser.add_argument('--data-dir', default='/home/dataset/tensorflow_datasets')
    parser.add_argument('--initial-lr', default=1e-3, type=float)
    parser.add_argument('--load_weights', action='store_true')
    parser.add_argument('--dryrun', action='store_true')

    parser.add_argument('--neg-ratio', default=3, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--log-interval', default=10, type=int)
    args = parser.parse_args()
    
    return args

def main():
    ''' main pipeline '''
    
    # parse arguments
    args = parse()
    
    # limit visible gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # create dataset
    logger.info("Creating dataset")
    train_dataset, test_dataset, num_train_samples, num_test_samples, labels = get_voc_dataset(args.batch_size)
    logger.info("Number of train samples: {},  Number of test samples: {},  Number of classes = {}".format(num_train_samples, num_test_samples, len(labels)-1))
    
    # get model
    if args.arch == "ssd_mobilenet2":
        from models.ssd_mobilenet2 import get_model, init_model
    else:
        from models.ssd_vgg16 import get_model, init_model

    hyper_params = utils_train.get_hyper_params(args.arch)
    hyper_params['total_labels'] = len(labels)
    img_size = hyper_params["img_size"]
    
    # create model
    ssd_model = get_model(hyper_params)
    ssd_custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
    ssd_model.compile(optimizer=Adam(learning_rate=args.initial_lr), loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
    init_model(ssd_model)
    
    ssd_model_path = utils_io.get_model_path(args.arch)
    if args.load_weights:
        ssd_model.load_weights(ssd_model_path)
    ssd_log_path = utils_io.get_log_path(args.arch)


    # We calculate prior boxes for one time and use it for all operations because all images are the same sizes
    prior_boxes = utils_bbox.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
    ssd_train_feed = utils_train.generator(train_dataset, prior_boxes, hyper_params)
    ssd_val_feed = utils_train.generator(test_dataset, prior_boxes, hyper_params)

    # callbacks during training
    checkpoint_callback = ModelCheckpoint(ssd_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
    tensorboard_callback = TensorBoard(log_dir=ssd_log_path)
    learning_rate_callback = LearningRateScheduler(utils_train.scheduler, verbose=0)

    # train
    step_size_train = utils_train.get_step_size(num_train_samples, args.batch_size)
    step_size_val = utils_train.get_step_size(num_test_samples, args.batch_size)
    ssd_model.fit(ssd_train_feed,
                  steps_per_epoch=step_size_train,
                  validation_data=ssd_val_feed,
                  validation_steps=step_size_val,
                  epochs=args.epochs,
                  callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback])


if __name__ == '__main__':
    main()