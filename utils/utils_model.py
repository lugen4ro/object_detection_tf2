from utils import utils_train, utils_io
from loss.ssd_loss import CustomLoss
from tensorflow.keras.optimizers import SGD, Adam
import logging
#logging.basicConfig(level=logging.ERROR, format='%(message)s') # ERROR to suppress deprecated warning for distorted boundig box message...
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

def get_model(arch, num_classes, weights):
    
    hyper_params = utils_train.get_hyper_params(arch)
    hyper_params['total_labels'] = num_classes
    img_size = hyper_params["img_size"]
    
    if arch == "SSD_VGG16":
        from models.ssd_vgg16 import SSD_VGG16
        ssd_model = SSD_VGG16(hyper_params)
    elif arch == "SSD_Mobilenet2":
        from models.ssd_mobilenet2 import SSD_Mobilenet2
        ssd_model = SSD_Mobilenet2(hyper_params)
    else:
        raise RuntimeError("Unkonwn Architecture Specified. ---> {}".format(arch))
    #init_model(ssd_model)
    
    #ssd_model_path = utils_io.get_model_path(arch)
    if weights:
        logger.info("Loading Weights from: {}".format(weights))
        ssd_model.load_weights(weights)
    
    return ssd_model, hyper_params
