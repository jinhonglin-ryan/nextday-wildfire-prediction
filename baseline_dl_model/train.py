import torch
import torch.nn as nn
import torch.optim as optim
from convlstm import ConvLSTM_Seg  # Your ConvLSTM segmentation model
from data_processing.modified.dataloader import load

# Configuration for your dataset and dataloader
config = {
    'dataset_directory': '/Users/erice/OneDrive/Desktop/nextday-wildfire-prediction/modified_dataset/ndws_western_dataset',
    'stats_directory': '/Users/erice/OneDrive/Desktop/nextday-wildfire-prediction/modified_dataset/data_statistics',
    'batch_size': 16,
    'features_to_drop': [],
    'rescale': False,         # Set False if you don't want to apply normalization
    'crop_augment': False,     # Adjust augmentations as needed
    'rot_augment': False,
    'flip_augment': False,
}

# Load the data loaders for 'train', 'test', and 'eval'
data_loaders = load(config)
train_loader = data_loaders['train']

# Initialize the ConvLSTM segmentation model
model = ConvLSTM_Seg(
    num_classes=2,         # For fire vs. no-fire segmentation
    input_size=(32, 32),   # Spatial size after augmentation (e.g., 32x32)
    input_dim=22,          # Number of feature channels
    hidden_dim=16,         # Number of channels in the hidden state
    kernel_size=(3, 3),
    pad_value=0
).cuda()  # Move to GPU if available

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        # data shape: (B, C, H, W) where C=number of features (here 22)
        # target shape: (B, H, W)
        data = data.cuda()
        target = target.long().cuda()  # CrossEntropyLoss expects target as LongTensor

        # Convert data to 5D tensor: add a time dimension (T)
        # Here we assume a single time-step, so T=1. If you have a sequence, adjust accordingly.
        data_5d = data.unsqueeze(1)  # Now shape: (B, 1, 22, H, W)

        # Forward pass through the model
        logits = model(data_5d)  # Expected output shape: (B, num_classes, H, W)

        # Compute loss
        loss = criterion(logits, target)

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / (batch_idx + 1)
    print(f"Epoch [{epoch+1}/{num_epochs}] complete. Average Loss: {avg_loss:.4f}")

print("Training complete!")
