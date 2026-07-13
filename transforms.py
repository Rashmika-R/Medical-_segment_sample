from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    DivisiblePadd,
    EnsureTyped,
)

# -------------------------------------------------
# Training Transforms
# -------------------------------------------------

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),

        EnsureChannelFirstd(keys=["image", "label"]),

        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
        ),

        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=300,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
        ),

        # Make every dimension divisible by 16
        DivisiblePadd(
            keys=["image", "label"],
            k=16,
        ),

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=1,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=0,
            prob=0.5,
        ),

        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            max_k=3,
        ),

        EnsureTyped(keys=["image", "label"]),
    ]
)

# -------------------------------------------------
# Validation Transforms
# -------------------------------------------------

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),

        EnsureChannelFirstd(keys=["image", "label"]),

        Orientationd(
            keys=["image", "label"],
            axcodes="RAS",
        ),

        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),

        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=300,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
        ),

        # IMPORTANT
        DivisiblePadd(
            keys=["image", "label"],
            k=16,
        ),

        EnsureTyped(keys=["image", "label"]),
    ]
)