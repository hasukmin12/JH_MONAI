import os, glob
import torch
import torchio as tio
from torch import optim
from pytorch_model_summary import summary

from loss_list import DiceCELoss_Portion, DiceFocalLoss_Portion


def call_only_model(args):
    model = None
    if args.MODEL_NAME in ['unetr', 'UNETR']:
        from monai.networks.nets import UNETR
        model = UNETR(
            in_channels = args.channel_in,
            out_channels = args.channel_out,
            img_size = args.input_shape,
            feature_size = args.patch_size,
            hidden_size = args.embed_dim,
            mlp_dim = args.mlp_dim,
            num_heads = args.num_heads,
            dropout_rate = args.dropout,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )
    elif args.MODEL_NAME in ['vnet', 'VNET', 'VNet', 'Vnet']:
        from monai.networks.nets import VNet
        model = VNet(
            spatial_dims=3,
            in_channels=args.channel_in,
            out_channels=args.channel_out,
        )
    
    elif args.MODEL_NAME in ['unet', 'UNET', 'UNet', 'Unet']:
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3,
            in_channels=args.channel_in,
            out_channels=args.channel_out,
            channels=(32, 64, 128, 256, 512),
            strides = (2, 2, 2, 2),
            dropout = 0.1,
            num_res_units=2,
        )

    
        # summary(model, torch.zeros((1,*args.input_shape)), show_input=True)
    # elif args.MODEL_NAME in ['swinUNETR', 'sunetr', 'sUNETR']:
    #     pass

    # optimizer = call_optimizer(args, model)
    # assert optimizer is not None, 'Optimization Error!'
    assert model is not None, 'Model Error!'
    
    return model.to(args.device) # , optimizer









def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

def call_subject_dataset(list_, transforms):
    subject_list = []
    for this_ in list_:
        subject_list.append(tio.Subject(
            image=tio.ScalarImage(this_["image"]),
            label=tio.LabelMap(this_["label"])
        ))
    return tio.SubjectsDataset(subject_list, transform=transforms)

def call_fold_dataset(list_, target_fold, total_folds=5):
    train, valid = [],[]
    count = 0
    for i in list_:
        count += 1
        if count == total_folds: count = 1
        if count == target_fold:
            valid.append(i)
        else:
            train.append(i)
    return train, valid

def call_optimizer(args, model):
    if args.Optim_NAME in ['SGD', 'sgd']:
        return optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum)
    elif args.Optim_NAME in ['ADAM', 'adam', 'Adam']:
        return optim.Adam(model.parameters(), lr=args.lr_init)
    elif args.Optim_NAME in ['ADAMW', 'adamw', 'AdamW', 'Adamw']:
        return optim.AdamW(model.parameters(), lr=args.lr_init, weight_decay=args.lr_decay)
    elif args.Optim_NAME in ['ADAGRAD', 'adagrad', 'AdaGrad']:
        return optim.Adagrad(model.parameters(), lr=args.lr_init, lr_decay=args.lr_decay)
    else:
        return None

def call_model(args):
    model = None
    if args.MODEL_NAME in ['unetr', 'UNETR']:
        from monai.networks.nets import UNETR
        model = UNETR(
            in_channels = args.channel_in,
            out_channels = args.channel_out,
            img_size = args.input_shape,
            feature_size = args.patch_size,
            hidden_size = args.embed_dim,
            mlp_dim = args.mlp_dim,
            num_heads = args.num_heads,
            dropout_rate = args.dropout,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )
    elif args.MODEL_NAME in ['vnet', 'VNET', 'VNet', 'Vnet']:
        from monai.networks.nets import VNet
        model = VNet(
            spatial_dims=3,
            in_channels=args.channel_in,
            out_channels=args.channel_out,
        )
    # elif args.MODEL_NAME in ['unet', 'UNET', 'UNet', 'Unet']:
    #     from monai.networks.nets import UNet
    #     model = UNet(
    #         spatial_dims=3,
    #         in_channels=args.channel_in,
    #         out_channels=args.channel_out,
    #         channels=(16, 32, 64, 128, 256),
    #         strides=(2, 2, 2, 2),
    #         num_res_units=2,
    #     )
    elif args.MODEL_NAME in ['unet', 'UNET', 'UNet', 'Unet']:
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3,
            in_channels=args.channel_in,
            out_channels=args.channel_out,
            channels=(32, 64, 128, 256, 512),
            strides = (2, 2, 2, 2),
            dropout = 0.1,
            num_res_units=2,
        )

    
        # summary(model, torch.zeros((1,*args.input_shape)), show_input=True)
    elif args.MODEL_NAME in ['swinUNETR', 'sunetr', 'sUNETR']:
        pass

    optimizer = call_optimizer(args, model)
    assert optimizer is not None, 'Optimization Error!'
    assert model is not None, 'Model Error!'
    
    return model.to(args.device), optimizer

import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, GeneralizedDiceLoss, DiceFocalLoss
class call_loss(nn.Module):  
    def __init__(self, 
                loss_mode, 
                include_background=False, 
                sigmoid=False, 
                softmax=False, 
                y_onehot=False):
        super().__init__()
        self.Dice = DiceLoss(
            include_background=include_background, 
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.GDice = GeneralizedDiceLoss(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            ) #GDiceLoss() # GDiceLossV2()
        self.DiceCE = DiceCELoss(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            ) #DC_and_CE_loss()
        self.DiceFocal = DiceFocalLoss(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.DiceCE_Portion = DiceCELoss_Portion(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.DiceFocal_Portion = DiceFocalLoss_Portion(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
        )


        self.Loss_NAME = loss_mode

    def forward(self, pred, target):
        if self.Loss_NAME in ['dice', 'Dice', 'DICE']:
            return self.Dice(pred, target)
        elif self.Loss_NAME in ['gdice', 'GDice', 'Gdice', 'gen_dice']:
            return self.GDice(pred, target)
        elif self.Loss_NAME in ['Dice+CrossEntropy', 'DiceCE', 'dice+ce', 'dice_ce']:
            return self.DiceCE(pred, target)
        elif self.Loss_NAME in ['DiceFocal', 'dicefocal', 'Dice+Focal', 'dice+focal', 'dice_focal']:
            return self.DiceFocal(pred, target)
        elif self.Loss_NAME in ['DiceCE_Portion', 'diceceportion', 'Dice+Ce+Portion', 'dice+ce+portion', 'dice_ce_portion']:
            return self.DiceCE_Portion(pred, target)
        elif self.Loss_NAME in ['DiceFocal_Portion', 'dicefocalportion', 'Dice+Focal+Portion', 'dice+focal+portion', 'dice_focal_portion']:
            return self.DiceFocal_Portion(pred, target)

 
        
        