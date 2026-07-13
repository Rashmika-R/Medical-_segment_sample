import torch
from tqdm import tqdm

from dataloader import get_dataloaders
from model import get_model

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

# ---------------------------------------
# Device
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------
# Load Data
# ---------------------------------------
train_loader, val_loader = get_dataloaders()

# ---------------------------------------
# Load Model
# ---------------------------------------
model = get_model().to(device)
print("Model created successfully!")

# ---------------------------------------
# Loss Function
# ---------------------------------------
loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
)

# ---------------------------------------
# Optimizer
# ---------------------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

# ---------------------------------------
# Dice Metric
# ---------------------------------------
dice_metric = DiceMetric(
    include_background=False,
    reduction="mean",
)

post_pred = AsDiscrete(argmax=True, to_onehot=5)
post_label = AsDiscrete(to_onehot=5)

# ---------------------------------------
# Training Settings
# ---------------------------------------
max_epochs = 10

best_metric = -1
best_metric_epoch = -1

# ============================================
# Training Loop
# ============================================

for epoch in range(max_epochs):

    print("-" * 60)
    print(f"Epoch {epoch + 1}/{max_epochs}")

    model.train()

    epoch_loss = 0

    progress_bar = tqdm(train_loader)

    for batch_data in progress_bar:

        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        progress_bar.set_description(
            f"Loss: {loss.item():.4f}"
        )

    epoch_loss /= len(train_loader)

    print(f"\nAverage Training Loss : {epoch_loss:.4f}")

    # ===================================
    # Validation
    # ===================================

    model.eval()

    with torch.no_grad():

        for val_data in val_loader:

            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            print("Validation image shape :", val_images.shape)

            val_outputs = model(val_images)

            val_outputs = [
                post_pred(i)
                for i in decollate_batch(val_outputs)
            ]

            val_labels = [
                post_label(i)
                for i in decollate_batch(val_labels)
            ]

            dice_metric(y_pred=val_outputs, y=val_labels)

        metric = dice_metric.aggregate().item()

        dice_metric.reset()

        print(f"Validation Dice : {metric:.4f}")

        if metric > best_metric:

            best_metric = metric
            best_metric_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                "best_model.pth"
            )

            print("✅ Best model saved!")

print("\nTraining Completed!")

print(f"Best Dice Score : {best_metric:.4f}")
print(f"Best Epoch      : {best_metric_epoch}")