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
from monai.metrics import *
import matplotlib.pyplot as plt
import yaml

from call import *
from utils import *


def main():
    parser = ap.ArgumentParser()

    # # 사용하고자 하는 GPU 넘버 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    ## data3
    parser.add_argument('--target', '-n', default='multi_organ', dest='TARGET_NAME', type=str)
    parser.add_argument('--save_name', default='unet_focal_patch_192', type=str)
    parser.add_argument('--loss', default='DiceFocal', dest='Loss_NAME', type=str)

    parser.add_argument('--channel_in', default=1, dest='channel_in', type=int)
    parser.add_argument('--channel_out', default=6, dest='channel_out', type=int)

    # visualize input data
    parser.add_argument('--visualize', default=True, type=bool)

    ## training
    parser.add_argument('--optimizer', default='AdamW', dest='Optim_NAME', type=str)
    parser.add_argument('--model', '-m', default='unet', dest='MODEL_NAME', type=str)
    parser.add_argument('--load_model', default='False', dest='load_model', type=str)
    parser.add_argument('--batch_size', default=4, dest='BATCH_SIZE', type=int)    

    parser.add_argument('--max_iterations', default=50000, dest='max_iterations', type=int)
    parser.add_argument('--eval_num', default=500, dest='eval_num', type=int)
    parser.add_argument('--samples', default=20, dest='samples_per_volume', type=int)

    parser.add_argument('--seeds', default=42, dest='seeds', type=int)
    parser.add_argument('--workers', default=4, dest='num_workers', type=int)
    parser.add_argument('--fold', '-f', default=4, dest='FOLD', type=int)
    parser.add_argument('--num_folds', default=5, dest='FOLDS', type=int)
    parser.add_argument('--spacing', default='1,1,1', dest='spacing', type=str)

    
    # Roi는 꼭 16의 배수로 해야한다.(DownConv 과정에서 절반씩 줄어드는데 100같은거 해버리면 채널 128될때쯤에 25가되서 13,14로 나뉘어서 에러남)
    # parser.add_argument('--input', default='96,96,96', dest='input_shape', type=str)
    parser.add_argument('--input', default='160,160,64', dest='input_shape', type=str)

    # UNETR의 경우 args.patch_size를 꼭 정의해줘야한다.
    # parser.add_argument('--patch', default=32, dest='patch_size', type=int)

    parser.add_argument('--patch', default=32, dest='patch_size', type=int)
    parser.add_argument('--mlp_dim', default=3072, dest='mlp_dim', type=int)
    parser.add_argument('--num_layers', default=12, dest='num_layers', type=int)
    parser.add_argument('--ext_layers', default='3,6,9,12', dest='ext_layers', type=str)
    parser.add_argument('--embed', default=768, dest='embed_dim', type=int)
    parser.add_argument('--num_heads', default=12, dest='num_heads', type=int)
    parser.add_argument('--dropout', default=0.1, dest='dropout', type=float)

    # # kipa -> (0,2000)
    # parser.add_argument('--a_min', default=0.0, type=float, help='a_min in ScaleIntensityRanged')
    # parser.add_argument('--a_max', default=2000.0, type=float, help='a_max in ScaleIntensityRanged')

    # multi_organ -> (-150, 300)
    parser.add_argument('--a_min', default=0.0, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=500.0, type=float, help='a_max in ScaleIntensityRanged')

    parser.add_argument('--lr', default=0.0005, dest='lr_init', type=float)
    parser.add_argument('--lr_decay', default=1e-5, dest='lr_decay', type=float)
    parser.add_argument('--momentum', default=0.9, dest='momentum', type=float)

    args = parser.parse_args()

    assert args.FOLDS+1 > args.FOLD, 'Check total # of folds and target fold'

    # args.spacing = [float(this_) for this_ in args.spacing.split(',')]
    # args.CONTRAST = [int(this_) for this_ in args.CONTRAST.split(',')]
    args.input_shape = [int(this_) for this_ in args.input_shape.split(',')]
    args.ext_layers = [int(this_) for this_ in args.ext_layers.split(',')]
    args.load_model = True if args.load_model in ['true', 'True'] else False

    if args.TARGET_NAME in ['kipa', 'KiPA', 'KIPA']:
        args.TARGET_NAME = 'kipa'
        args.root = '/nas3/sukmin/dataset/Task302_KiPA'
        args.class_names = {1:'Vein', 2:'Kidney', 3:'Artery', 4:'Tumor'}  
    elif args.TARGET_NAME in ['rib', 'Rib', 'RIB']:
        args.TARGET_NAME = 'rib'
        args.root = '/disk1/MIAI/labels_rib'
        args.class_names = {1: args.TARGET_NAME}
    elif args.TARGET_NAME in ['skin', 'Skin', 'SKIN']:
        args.TARGET_NAME = 'skin'
        args.root = '/disk1/MIAI/labels_skin'
        args.class_names = {1: args.TARGET_NAME}
    elif args.TARGET_NAME in ['awall', 'abdwall', 'Awall', 'AWall']:
        args.TARGET_NAME = 'awall'
        args.root = '/disk1/MIAI/labels_awall'
        args.class_names = {1: args.TARGET_NAME}
    # elif args.TARGET_NAME in ['organs', 'Organs', 'organ', 'Oran']:
    #     args.TARGET_NAME = 'organ'
    #     args.class_names = {1:'Liver', 2:'Stomach', 3:'Pancreas', 4:'Gallbladder', 5:'Spleen'} 
    elif args.TARGET_NAME in ['Multi_Organ', 'multiorgan', 'multi_organ']:
        args.TARGET_NAME = 'multi_organ'
        args.root = '/nas3/sukmin/dataset/Task002_Multi_Organ'
        args.class_names = {1:'Liver', 2:'Stomach', 3:'Pancreas', 4:'Gallbladder', 5:'Spleen'} 


    else:
        print('Wrong target name!')

    # args.LOGDIR = f'/disk1/sukmin/{args.MODEL_NAME}/{args.TARGET_NAME}/fold{args.FOLD}'
    args.LOGDIR = f'/nas3/sukmin/{args.TARGET_NAME}_model/{args.save_name}'
    
    if not os.path.isdir(args.LOGDIR):
        os.makedirs(args.LOGDIR, exist_ok=True)

    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(args.seeds)       # python random seed
    torch.manual_seed(args.seeds) # pytorch random seed
    np.random.seed(args.seeds)    # numpy random seed
    torch.backends.cudnn.deterministic = True


    


    # run
    run = wandb.init() # project="has_kipa", entity="hutom_miai") 
    wandb.config.update(args)

    # Dataset
    datasets = os.path.join(args.root, 'dataset.json')
    file_list = load_decathlon_datalist(datasets, True, 'training')
    train_list, valid_list = call_fold_dataset(file_list, target_fold=args.FOLD, total_folds=args.FOLDS)    
    print('Train', len(train_list), 'Valid', len(valid_list))
    artifact = wandb.Artifact(
        "dataset", type="dataset", 
        metadata={"train_list":train_list, "valid_list":valid_list, "train_len":len(train_list), "valid_len":len(valid_list)})
    run.log_artifact(artifact)

    if args.TARGET_NAME == 'kipa':
        from trans_kipa import call_transforms
        train_transforms, val_transforms = call_transforms(args)
    elif args.TARGET_NAME == 'multi_organ':
        from trains_multi_organ import call_transforms
        train_transforms, val_transforms = call_transforms(args)

    # elif args.TARGET_NAME == 'awall':
    #     from trans_awall import train_transforms, val_transforms
    #     train_transforms, val_transforms = call_transforms(args)
    # if args.TARGET_NAME == 'rib':
    #     from trans_rib import train_transforms, val_transforms
    #     train_transforms, val_transforms = call_transforms(args)

    args.num_workers = torch.cuda.device_count() * 4
    
    train_ds = CacheDataset(
        data=train_list,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=args.num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=4, num_workers=args.num_workers, 
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

    # Initialize model, optimizer and loss function
    model, optimizer = call_model(args)
    # model = nn.DataParallel(model)
    # net = nn.DataParallel(netG, device_ids=list(range(NGPU
    # model.to(device)
    print(model)
    

    if args.visualize == True:
        # slice_map = {
        #     "case_00001.nii.gz": 40,
        #     "case_00002.nii.gz": 50,
        #     "case_00003.nii.gz": 60,
        # }
        case_num = 0
        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        img_shape = img.shape
        label_shape = label.shape
        print(f"image shape: {img_shape}, label shape: {label_shape}")
        # plt.figure("image", (18, 6))
        # plt.subplot(1, 2, 1)
        # plt.title("image")
        # plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.title("label")
        # plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
        # plt.show()


    if args.load_model:
        try:
            checkpoint = torch.load(os.path.join(args.LOGDIR, f"model_best.pth"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            model.load_state_dict(torch.load(os.path.join(args.LOGDIR, f"model_best.pth")))
        print('Model Loaded!')

    wandb.watch(model, log="all")

    best_loss = 1.
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    while global_step < args.max_iterations:
        global_step, dice_val_best, global_step_best = train(
            args, run, model, optimizer, global_step, train_loader, val_loader, dice_val_best, global_step_best
        )

    with open(os.path.join(args.LOGDIR, 'config.yml'),'w') as f:
        f.write(yaml.dump(args))
        f.close()



def validation(args, epoch_iterator_val, model, threshold=0.5):
    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!
    dice_metric = DiceMetric(include_background=False, reduction='none')
    confusion_matrix = ConfusionMatrixMetric(include_background=False, reduction='none')

    dice_class, mr_class, fo_class = [], [], []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            step += 1
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            
            val_outputs = activation(sliding_window_inference(val_inputs, args.input_shape, 4, model))

            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )
            dice_class.append(dice_metric(val_outputs>=threshold, val_labels)[0])

            confusion = confusion_matrix(val_outputs>=threshold, val_labels)[0]
            mr_class.append([
                calc_confusion_metric('fnr',confusion[i]) for i in range(args.channel_out-1)
            ])
            fo_class.append([
                calc_confusion_metric('fpr',confusion[i]) for i in range(args.channel_out-1)
            ])
        dice_dict, dice_val = calc_mean_class(args, dice_class, 'valid_dice')
        miss_dict, miss_val = calc_mean_class(args, mr_class, 'valid_miss rate')
        false_dict, false_val = calc_mean_class(args, fo_class, 'valid_false alarm')

        wandb.log({
            'valid_dice': dice_val,
            'valid_miss rate': miss_val,
            'valid_false alarm': false_val,
            'valid_image': log_image_table(args, val_inputs[0].cpu(),
                                            val_labels[0].cpu(),val_outputs[0].cpu()),
        })
        wandb.log(dice_dict)
        wandb.log(miss_dict)
        wandb.log(false_dict)
    return dice_val


def train(args, run, model, optimizer, global_step, train_loader, val_loader, dice_val_best, global_step_best):  
    loss_function = call_loss(loss_mode = args.Loss_NAME, sigmoid=True)
    dice_loss = call_loss(loss_mode = 'dice', sigmoid=True)

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

        loss = loss_function(logit_map, y)
        dice = 1 - dice_loss(logit_map, y)

        epoch_loss += loss.item()
        epoch_dice += dice.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step+1, args.max_iterations, loss)
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
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(args.LOGDIR, f"model_best.pth"))

                artifact = wandb.Artifact('model', type='model')
                artifact.add_file(
                    os.path.join(args.LOGDIR, f"model_best.pth"), 
                    name=f'model/{args.MODEL_NAME}')
                run.log_artifact(artifact)
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



if __name__ == "__main__":
    main()
    wandb.finish()