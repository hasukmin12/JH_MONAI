import os, tempfile
import random
import torch
import wandb
import argparse as ap
import numpy as np
from tqdm import tqdm

from monai.engines import create_multigpu_supervised_trainer
from monai.inferers import sliding_window_inference
from monai.data import *
from monai.transforms import *
from monai.metrics import DiceMetric

from call import *


def log_image_table(args, image, label, predict):
    mask_images = []
    image = image[0]
    label = torch.argmax(label, dim=0) if not label.size()[0] == 1 else label[0]
    predict = torch.argmax(predict, dim=0) if not predict.size()[0] == 1 else predict[0]

    frames = int(np.round(image.shape[-1]/3))
    for frame in range(frames,frames*2,2):
        mask_images.append(wandb.Image(image[...,frame].numpy(), masks={
            "ground_truth":{"mask_data":label[...,frame].numpy(),"class_labels":args.class_names},
            "predictions":{"mask_data":predict[...,frame].numpy(),"class_labels":args.class_names},
        }))
    return mask_images

def validation(args, epoch_iterator_val, model):
    dice_loss = call_loss('dice', y_onehot=True, softmax=True, sigmoid=False)
    dice = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            step += 1
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            
            val_outputs = sliding_window_inference(val_inputs, args.input_shape, 4, model)

            dice += 1 - dice_loss(val_outputs, val_labels)

            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )
        wandb.log({
            'valid_dice': dice / step,
            'valid_image': log_image_table(args, val_inputs[0].cpu(),
                                            val_labels[0].cpu(),val_outputs[0].cpu()),
        })
    return dice / step


def train(args, global_step, train_loader, val_loader, dice_val_best, global_step_best):
    # Initialize model, optimizer and loss function
    model, optimizer = call_model(args)
    loss_function = call_loss(loss_mode = args.Loss_NAME, y_onehot=True, softmax=True, sigmoid=False)
    dice_loss = call_loss(loss_mode = 'dice', y_onehot=True, softmax=True, sigmoid=False)

    wandb.watch(model, log="all")

    model.train()

    step = 0
    epoch_loss, epoch_dice = 0., 0.
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        
        logit_map = model(x)
        print('shape', x.size(), y.size(), logit_map.size())
        loss = loss_function(logit_map, y)
        dice = 1 - dice_loss(logit_map, y)
        print('loss size', loss.size(), dice.size())
        epoch_loss += loss.item()
        epoch_dice += dice.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.max_iterations, loss)
        )

        if (
            global_step % args.eval_num == 0 #and global_step != 0
        ) or global_step == args.max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps)", dynamic_ncols=True
            )
            dice_val = validation(args, epoch_iterator_val, model)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(args.LOGDIR, f"model_{global_step}.pth")
                )
                print(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )
            else:
                print(
                    f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )
        global_step += 1
    wandb.log({
        'train_loss': epoch_loss / step,
        'train_dice': epoch_dice / step,
    })
    return global_step, dice_val_best, global_step_best

def main(args):
    # Dataset
    datasets = os.path.join(args.root, 'dataset_monai.json')
    file_list = load_decathlon_datalist(datasets, True, 'training')
    train_list, valid_list = call_fold_dataset(file_list, target_fold=args.FOLD, total_folds=args.FOLDS)    
    print('Train', len(train_list), 'Valid', len(valid_list))

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # ScaleIntensityRanged(keys=["image"], 
            #     a_min=args.CONTRAST[0], a_max=args.CONTRAST[1], 
            #     b_min=-1, b_max=1, clip=True),
            NormalizeIntensityd(keys=["image"]),
            RandCropByPosNegLabeld(
                keys=["image","label"], 
                label_key="label",
                num_samples=2,
                spatial_size=args.input_shape, 
                ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # ScaleIntensityRanged(keys=["image"], 
            #     a_min=args.CONTRAST[0], a_max=args.CONTRAST[1], 
            #     b_min=-1, b_max=1, clip=True),
            NormalizeIntensityd(keys=["image"]),
            ToTensord(keys=["image", "label"]),
        ]
    )
    
    train_ds = CacheDataset(
        data=train_list,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=1, num_workers=args.num_workers, 
        pin_memory=True, #shuffle=True, 
    )
    val_ds = CacheDataset(
        data=valid_list, 
        transform=val_transforms, 
        cache_rate=1.0, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=args.num_workers, 
        pin_memory=True, #shuffle=False, 
    )
    best_loss = 1.
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    while global_step < args.max_iterations:
        global_step, dice_val_best, global_step_best = train(
            args, global_step, train_loader, val_loader, dice_val_best, global_step_best
        )

def get_arguments():
    parser = ap.ArgumentParser()
    ## data
    parser.add_argument('--target', '-n', default=None, dest='TARGET_NAME', type=str)
    parser.add_argument('--fold', '-f', default=None, dest='FOLD', type=int)
    parser.add_argument('--num_folds', default=5, dest='FOLDS', type=int)
    parser.add_argument('--spacing', default='1,1,1', dest='spacing', type=str)
    parser.add_argument('--window', default='-150,300', dest='CONTRAST', type=str)
    parser.add_argument('--channel_in', default=1, dest='channel_in', type=int)
    parser.add_argument('--channel_out', default=1, dest='channel_out', type=int)

    ## training
    parser.add_argument('--batch_size', default=4, dest='BATCH_SIZE', type=int)    
    parser.add_argument('--max_iterations', default=25000, dest='max_iterations', type=int)
    parser.add_argument('--eval_num', default=500, dest='eval_num', type=int)
    parser.add_argument('--samples', default=20, dest='samples_per_volume', type=int)

    parser.add_argument('--seeds', default=42, dest='seeds', type=int)
    parser.add_argument('--workers', default=6, dest='num_workers', type=int)

    ## transformer
    parser.add_argument('--optimizer', default='SGD', dest='Optim_NAME', type=str)
    parser.add_argument('--model', '-m', default='unetr', dest='MODEL_NAME', type=str)
    parser.add_argument('--input', default='96,96,96', dest='input_shape', type=str)
    parser.add_argument('--patch', default=32, dest='patch_size', type=int)
    parser.add_argument('--mlp_dim', default=3072, dest='mlp_dim', type=int)
    parser.add_argument('--num_layers', default=12, dest='num_layers', type=int)
    parser.add_argument('--ext_layers', default='3,6,9,12', dest='ext_layers', type=str)
    parser.add_argument('--embed', default=768, dest='embed_dim', type=int)
    parser.add_argument('--num_heads', default=12, dest='num_heads', type=int)
    parser.add_argument('--dropout', default=0.0, dest='dropout', type=float)

    parser.add_argument('--loss', default='dice', dest='Loss_NAME', type=str)
    parser.add_argument('--lr', default=0.0005, dest='lr_init', type=float)
    parser.add_argument('--lr_decay', default=1e-5, dest='lr_decay', type=float)
    parser.add_argument('--momentum', default=0.9, dest='momentum', type=float)

    args = parser.parse_args()
    assert args.FOLDS+1 > args.FOLD, 'Check total # of folds and target fold'

    args.spacing = [float(this_) for this_ in args.spacing.split(',')]
    args.CONTRAST = [int(this_) for this_ in args.CONTRAST.split(',')]
    args.input_shape = [int(this_) for this_ in args.input_shape.split(',')]
    args.ext_layers = [int(this_) for this_ in args.ext_layers.split(',')]

    if args.TARGET_NAME in ['organs', 'Organs', 'organ', 'Oran']:
        args.TARGET_NAME = 'organ'
        args.class_names = {1:'Liver', 2:'Stomach', 3:'Pancreas', 4:'Gallbladder', 5:'Spleen'}
    elif args.TARGET_NAME in ['rib', 'Rib', 'RIB']:
        args.TARGET_NAME = 'rib'
        args.root = ''
        args.class_names = {1: args.TARGET_NAME}
    elif args.TARGET_NAME in ['skin', 'Skin', 'SKIN']:
        args.TARGET_NAME = 'skin'
        args.root = ''
        args.class_names = {1: args.TARGET_NAME}
    elif args.TARGET_NAME in ['awall', 'abdwall', 'Awall', 'AWall']:
        args.TARGET_NAME = 'awall'
        args.root = ''
        args.class_names = {1: args.TARGET_NAME}
    elif args.TARGET_NAME in ['kipa', 'KiPA', 'KIPA']:
        args.TARGET_NAME = 'kipa'
        args.root = '/disk1/KiPA2022/train'
        args.class_names = {1:'1', 2:'2', 3:'3', 4:'4'}        
    else:
        print('Wrong target name!')

    args.LOGDIR = f'/disk1/jepark/{args.MODEL_NAME}/{args.TARGET_NAME}/fold{args.FOLD}'
    if os.path.isdir(args.LOGDIR):
        os.system(f'rm -rf {args.LOGDIR}')
    os.makedirs(args.LOGDIR, exist_ok=True)

    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(args.seeds)       # python random seed
    torch.manual_seed(args.seeds) # pytorch random seed
    np.random.seed(args.seeds)    # numpy random seed
    torch.backends.cudnn.deterministic = True
    return args

if __name__ == "__main__":
    args = get_arguments()

    wandb.init(project=f'{args.TARGET_NAME}', entity='jeune') 
    wandb.config.update(args)
    main(args)
    
    with open(os.path.join(args.LOGDIR, 'config.yml'),'w') as f:
        f.write(yaml.dump(args))
        f.close()
