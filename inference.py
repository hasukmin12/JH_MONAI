import glob
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
import yaml

from call import *
from utils import *



def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--data_dir', '-i', default="/disk1/sukmin/dataset/Task302_KiPA", type=str)
    parser.add_argument('--output_dir', '-o', default="/disk1/sukmin/inf_rst", type=str)
    parser.add_argument('--pth_path', default="/disk1/sukmin/unet/kipa/fold4/model_best.pth", type=str)
    parser.add_argument('--model', '-m', default='unet', dest='MODEL_NAME', type=str)

    # parser.add_argument('--input', default='96,96,96', dest='input_shape', type=str)

    # default
    # parser.add_argument('--window', default = '0,300',  dest='CONTRAST', type=str)# default='-150,300', dest='CONTRAST', type=str)
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--channel_in', default=1, dest='channel_in', type=int)
    parser.add_argument('--channel_out', default=5, dest='channel_out', type=int)
    parser.add_argument('--dropout', default=0.0, dest='dropout', type=float)
    parser.add_argument('--optimizer', default='AdamW', dest='Optim_NAME', type=str)
    parser.add_argument('--lr', default=0.0005, dest='lr_init', type=float)
    parser.add_argument('--lr_decay', default=1e-5, dest='lr_decay', type=float)
    parser.add_argument('--a_min', default=0.0, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=2000.0, type=float, help='a_max in ScaleIntensityRanged')
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = '1'


    print(args.output_dir)
    print(args.MODEL_NAME)

    roi_size = [args.roi_x, args.roi_y, args.roi_z]
    
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")
    rst_dir = os.path.join(args.output_dir, args.MODEL_NAME)
    if os.path.isdir(rst_dir)== False:
        os.makedirs(rst_dir)

    data_dir = args.data_dir


    test_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))

    test_data = [{"image": image} for image in test_images]


    test_org_transforms = Compose(
        [

            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRanged(keys=["image"], 
                a_min=args.a_min, a_max=args.a_max, 
                b_min=0, b_max=1, clip=True),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys="image"),

            # LoadImaged(keys=["image", "label"]),
            # EnsureChannelFirstd(keys=["image", "label"]),
            # AsDiscreted(keys=["label"], to_onehot=args.channel_out),
            # ScaleIntensityRanged(keys=["image"], 
            #     a_min=args.a_min, a_max=args.a_max, 
            #     b_min=0, b_max=1, clip=True),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # ToTensord(keys=["image", "label"]),
        ]
    )

    test_org_ds = Dataset(
        data=test_data, transform=test_org_transforms)

    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

    # post_transforms = Compose(
    #     EnsureTyped(keys="pred"),
    #     Invertd(
    #         keys="pred",
    #         transform=test_org_transforms,
    #         orig_keys="image",
    #         meta_keys="pred_meta_dict",
    #         orig_meta_keys="image_meta_dict",
    #         meta_key_postfix="meta_dict",
    #         nearest_interp=False,
    #         to_tensor=True,
    #     ),
    #     AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    #     SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=rst_dir, output_postfix="seg", resample=False),
    
    # post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    post_trans = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=0.5)])

    saver = SaveImage(output_dir=rst_dir, output_ext=".nii.gz", output_postfix="seg")
    
    # saver = SaveImage(keys='data', output_dir=rst_dir,  output_ext=".nii.gz", output_postfix="seg")
    



    model = call_only_model(args)
    # print(model)


    model.load_state_dict(torch.load(args.pth_path)["model_state_dict"])
        # os.path.join(args.pth_path, "best_metric_model.pth")))
    model.eval()


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")



    from monai.handlers.utils import from_engine
    from monai.metrics import DiceMetric
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    import nibabel as nib
    from monai.data import nifti_saver


    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!
    dice_metric = DiceMetric(include_background=False, reduction='none')
    confusion_matrix = ConfusionMatrixMetric(include_background=False, reduction='none')
    threshold=0.5

    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            sw_batch_size = 4
            # test_name = test_data['image_meta_dict']['filename_or_obj'][0][-17:]

            # test_output = activation(sliding_window_inference(test_inputs, inf_size, 4, model))
            pred_data = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            test_outputs = [post_trans(i) for i in decollate_batch(pred_data)]

            # for test_output in test_outputs:
            #     saver(test_output)

            meta_data = decollate_batch(test_data["image_meta_dict"])
            for test_output, data in zip(test_outputs, meta_data):
                saver(test_output, data)
            


if __name__ == "__main__":
    main()
