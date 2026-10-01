import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from src.dataset import ACDC2DDataset
from src.model import SharedEncoderMultiTaskNet
from src.loss import DirectionalGradientLoss

def train_epoch():
    device = torch.device(config.DEVICE)
    print(f"Initializing training on device: {device}")

    # Dataset & Dataloader
    dataset = ACDC2DDataset()
    if len(dataset) == 0:
        print("[WARNING] No dataset path found. Executing 1 dry-run batch with synthetic data...")
        images = torch.randn(config.BATCH_SIZE, 1, *config.TARGET_SIZE).to(device)
        masks = torch.randint(0, config.NUM_SEG_CLASSES, (config.BATCH_SIZE, *config.TARGET_SIZE)).to(device)
        clinical_classes = torch.randint(0, config.NUM_DISEASE_CLASSES, (config.BATCH_SIZE,)).to(device)
        train_loader = [(images, masks, clinical_classes)]
    else:
        train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize Architecture & Criteria
    model = SharedEncoderMultiTaskNet().to(device)
    seg_criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()
    topo_criterion = DirectionalGradientLoss().to(device)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    for batch_idx, (images, masks, clinical_classes) in enumerate(train_loader):
        images, masks, clinical_classes = images.to(device), masks.to(device), clinical_classes.to(device)

        optimizer.zero_grad()
        mask_preds, class_preds = model(images)

        L_seg = seg_criterion(mask_preds, masks)
        L_class = class_criterion(class_preds, clinical_classes)
        L_topo = topo_criterion(mask_preds, masks)

        L_total = (config.ALPHA * L_seg) + (config.BETA * L_class) + (config.GAMMA * L_topo)

        L_total.backward()
        optimizer.step()

        print(f"Batch {batch_idx:03d} | L_seg: {L_seg.item():.3f} | L_class: {L_class.item():.3f} | L_topo: {L_topo.item():.3f} | L_total: {L_total.item():.3f}")

    print("Success! Epoch dry-run completed.")

if __name__ == "__main__":
    train_epoch()
