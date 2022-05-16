from monai.transforms import *

def call_transforms(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            AsDiscreted(keys=["label"], to_onehot=args.channel_out),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ScaleIntensityRanged(keys=["image"], 
                a_min=args.CONTRAST[0], a_max=args.CONTRAST[1],
                b_min=0, b_max=1, clip=True),
            RandCropByPosNegLabeld(
                keys=["image","label"], 
                label_key="label",
                num_samples=2,
                spatial_size=args.input_shape, 
                ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            AsDiscreted(keys=["label"], to_onehot=args.channel_out),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ScaleIntensityRanged(keys=["image"], 
                a_min=args.CONTRAST[0], a_max=args.CONTRAST[1],
                b_min=0, b_max=1, clip=True),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms