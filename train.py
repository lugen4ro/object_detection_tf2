import argsparse
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # limit tensorflow output to terminal


def parse():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='ssd300')
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--pretrained-type', default='base')

    parser.add_argument('--neg-ratio', default=3, type=int)
    parser.add_argument('--initial-lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--checkpoint-dir', default='/home/work2/weights/tf_checkpoints')
    parser.add_argument('--pretrained-type', default='base')
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

    
    if args.arch == 'ssd300'



    
    
    
    
    
    
    

#
if backbone == "mobilenet_v2":
    from models.ssd_mobilenet_v2 import get_model, init_model
else:
    from models.ssd_vgg16 import get_model, init_model
#
hyper_params = train_utils.get_hyper_params(backbone)
#
train_data, info = data_utils.get_dataset("voc/2007", "train+validation", data_dir="/home/dataset/tensorflow_datasets")
val_data, _ = data_utils.get_dataset("voc/2007", "test", data_dir="/home/dataset/tensorflow_datasets")
train_total_items = data_utils.get_total_item_size(info, "train+validation")
val_total_items = data_utils.get_total_item_size(info, "test")

if with_voc_2012:
    voc_2012_data, voc_2012_info = data_utils.get_dataset("voc/2012", "train+validation", data_dir="/home/dataset/tensorflow_datasets")
    voc_2012_total_items = data_utils.get_total_item_size(voc_2012_info, "train+validation")
    train_total_items += voc_2012_total_items
    train_data = train_data.concatenate(voc_2012_data)

labels = data_utils.get_labels(info)
labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, augmentation.apply))
val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()
train_data = train_data.shuffle(batch_size*4).padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
val_data = val_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)
#
ssd_model = get_model(hyper_params)
ssd_custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
ssd_model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
init_model(ssd_model)
#
ssd_model_path = io_utils.get_model_path(backbone)
if load_weights:
    ssd_model.load_weights(ssd_model_path)
ssd_log_path = io_utils.get_log_path(backbone)
# We calculate prior boxes for one time and use it for all operations because of the all images are the same sizes
prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_train_feed = train_utils.generator(train_data, prior_boxes, hyper_params)
ssd_val_feed = train_utils.generator(val_data, prior_boxes, hyper_params)

checkpoint_callback = ModelCheckpoint(ssd_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=ssd_log_path)
learning_rate_callback = LearningRateScheduler(train_utils.scheduler, verbose=0)

step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)
ssd_model.fit(ssd_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=ssd_val_feed,
              validation_steps=step_size_val,
              epochs=epochs,
              callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback])


if __name__ == '__main__':
    main()