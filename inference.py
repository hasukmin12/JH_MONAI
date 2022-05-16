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


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_arguments():
    parser = ap.ArgumentParser()

    parser.add_argument('--data_dir', '-i', default="/disk1/sukmin/dataset/Task302_KiPA", type=str)
    parser.add_argument('--target_dir', '-o', default="/disk1/sukmin/rst/", type=str)
    parser.add_argument('--pth_path', '-o', default="/disk1/sukmin/unet/kipa/fold4/model_best.pth", type=str)
    parser.add_argument('--model', '-m', default='unet', dest='MODEL_NAME', type=str)
    parser.add_argument('--input', default='96,96,96', dest='input_shape', type=str)
    args = parser.parse_args()

args = get_arguments()

rst_dir = os.path.join(args.target_dir, args.model)
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
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(
            1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-57, a_max=164,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys="image"),
    ]
)

test_org_ds = Dataset(
    data=test_data, transform=test_org_transforms)

test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=test_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out", output_postfix="seg", resample=False),
])



model, optimizer = call_model(args)


model.load_state_dict(torch.load(
    args.pth_path))
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


metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []


with torch.no_grad():
    for test_data in test_org_loader:
        test_inputs = test_data["image"].to(device)
        roi_size = args.input_shape
        sw_batch_size = 4
        test_data["pred"] = sliding_window_inference(
            test_inputs, roi_size, sw_batch_size, model)

        # output
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]



        test_outputs, test_labels = from_engine(["pred", "label"])(test_data)
        dice_metric(y_pred=test_outputs, y=test_labels)
        dice_metric_batch(y_pred=test_outputs, y=test_labels)


        # data_dict["image_meta_dict"]["original_affine"] = None
        saver = NiftiSaver(output_dir=rst_dir, output_postfix="test", output_ext=".nii.gz", mode="nearest")
        saver.save(test_outputs)# data_dict["image"], data_dict["image_meta_dict"])    


        # img_case = os.path.join(rst_dir, test_outputs)
        # xform = np.eye(4) * 2
        # img_Nifti = nib.nifti1.Nifti1Image(test_outputs, xform)
        # nib.save(img_Nifti, img_case)




    metric = dice_metric.aggregate().item()
    metric_batch = dice_metric_batch.aggregate()

    # metric = dice_metric.aggregate().item()
    # metric_values.append(metric)
    # metric_batch = dice_metric_batch.aggregate()
    # metric_tc = metric_batch[0].item()
    # metric_values_tc.append(metric_tc)
    # metric_wt = metric_batch[1].item()
    # metric_values_wt.append(metric_wt)
    # metric_et = metric_batch[2].item()
    # metric_values_et.append(metric_et)

    dice_metric.reset()
    dice_metric_batch.reset()

metric_1, metric_2, metric_3, metric_4 = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item(), metric_batch[3].item()

print("Metric on original image spacing: ", metric)
print(f"metric_1: {metric_1:.4f}")
print(f"metric_2: {metric_2:.4f}")
print(f"metric_3: {metric_3:.4f}")
print(f"metric_4: {metric_4:.4f}")




#         # uncomment the following lines to visualize the predicted results
#         test_output = from_engine(["pred"])(test_data)

#         original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]

#         plt.figure("check", (18, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(original_image[:, :, 20], cmap="gray")
#         plt.subplot(1, 2, 2)
#         plt.imshow(test_output[0].detach().cpu()[1, :, :, 20])
#         plt.show()