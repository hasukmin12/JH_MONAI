from monai.transforms import *
def call_transforms(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            AsDiscreted(keys=["label"], to_onehot=args.channel_out),
            ScaleIntensityRanged(keys=["image"], 
                a_min=args.a_min, a_max=args.a_max, 
                b_min=0, b_max=1, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),

            # # Add Spacing
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     mode = ("bilinear", "nearest")
            # ),
            # NormalizeIntensityd(keys ="image", nonzero=True, channel_wise=True),


            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[0],
            #     prob=0.10,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[1],
            #     prob=0.10,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[2],
            #     prob=0.10,
            # ),
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.10,
            #     max_k=3,
            # ),
            # RandZoomd(
            #     keys=["image", "label"],
            #     prob=0.20,
            #     min_zoom=0.8,
            #     max_zoom=1.2,
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.50,
            # ),
            # RandHistogramShiftd(
            #     keys=["image"],
            #     num_control_points=10,
            #     prob=0.30,
            # ),
            RandCropByPosNegLabeld(
                keys=["image","label"], 
                label_key="label",
                pos=1,
                neg=1,
                num_samples=2,
                spatial_size=args.input_shape, 
                image_key="image",
                image_threshold=0,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            AsDiscreted(keys=["label"], to_onehot=args.channel_out),
            # ScaleIntensityRanged(keys=["image"], 
            #     a_min=args.a_min, a_max=args.a_max, 
            #     # a_min=args.CONTRAST[0], a_max=args.CONTRAST[1], 
            #     b_min=0, b_max=1, clip=True),
            # # Add Spacing
            # Spacingd(
            #     keys=["image", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     mode = ("bilinear", "nearest")
            # ),
            # NormalizeIntensityd(keys ="image", nonzero=True, channel_wise=True),
            
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms