
import *

def get_model(arch, pretrained_type, neg_ratio):
    
    import models
    
    
    model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    