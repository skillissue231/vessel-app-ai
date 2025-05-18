import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from model.unet import get_unet_model
from model.losses import DiceLoss
from utils.dataset import CAMDataset
from utils.transforms import get_transforms
from torch.amp import autocast, GradScaler

# Enable cuDNN autotuner for optimal performance
dbn = cudnn.enabled
cudnn.enabled = True
cudnn.benchmark = True


def train_model(config, dataset_path="data/cam", checkpoint_path="checkpoints", resume_model=None):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    print(f"Using device: {device}")

    # Initialize model with Attention U-Net
    model = get_unet_model(
        encoder_name=config['encoder'],
        num_classes=1,
        pretrained=config.get('pretrained', True),
        attention_type="scse"
    ).to(device)

    # Freeze encoder layers if fine-tuning except layer4
    if config.get('fine_tune', False):
        print("🔒 Freezing encoder layers for decoder-only fine-tuning (except layer4)")
        for name, param in model.named_parameters():
            if name.startswith('encoder') and 'layer4' not in name:
                param.requires_grad = False

    # Prepare optimizer with only trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(
        trainable_params,
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # Resume logic: attempt last valid checkpoint
    start_epoch = 0
    checkpoint_dir = os.path.join(checkpoint_path, config['encoder'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Gather only epoch_*.pth files
    ckpts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].split('.')[0]),
        reverse=True
    )
    resumed = False
    for ckpt_name in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        try:
            print(f"Attempting to resume from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                if not config.get('fine_tune', False):
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt.get('epoch', 0)
            else:
                model.load_state_dict(ckpt)
                start_epoch = int(ckpt_name.split('_')[-1].split('.')[0])
            print(f"  → Resumed at epoch {start_epoch+1}")
            resumed = True
            break
        except Exception as e:
            print(f"⚠️ Failed to load {ckpt_path}: {e}")
    if not resumed:
        print("No valid checkpoint found; starting from scratch.")

    # Dataset & DataLoader with safe settings
    train_ds = CAMDataset(dataset_path, transform=get_transforms())
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,              # Single worker for Colab compatibility
        pin_memory=False,           # Disable pinned memory
        prefetch_factor=None,       # No prefetch when num_workers=0
        persistent_workers=False    # Disable persistent workers
    )

    # Training parameters
    accumulation_steps = config.get('accumulation_steps', 1)
    total_epochs = start_epoch + config['epochs']

    # Scheduler: plateau for fine-tune else OneCycle
    if config.get('fine_tune', False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        total_steps = (len(train_loader) * total_epochs) // accumulation_steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            total_steps=total_steps
        )

    scaler = GradScaler()

    # Main training loop
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for step, (imgs, masks) in enumerate(pbar, 1):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast(device_type=device_type):
                outputs = model(imgs)
                loss = nn.BCEWithLogitsLoss()(outputs, masks) + DiceLoss()(outputs, masks)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if step % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            if not config.get('fine_tune', False):
                scheduler.step()
            pbar.set_postfix({'loss': running_loss / step})

        if config.get('fine_tune', False):
            scheduler.step(running_loss)

        # Save checkpoint
        ckpt = {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(ckpt, save_path)
        print(f"✔️ Saved checkpoint: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Attention U-Net with ResNet-34 on A100")
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, default='data/cam')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('--resume_model', type=str, default=None)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--unfreeze_encoder_layer4', action='store_true')
    args = parser.parse_args()

    cfg = {
        'encoder':            args.encoder,
        'pretrained':         True,
        'epochs':             args.epochs,
        'batch_size':         args.batch_size,
        'accumulation_steps': args.accumulation_steps,
        'lr':                 args.lr,
        'weight_decay':       args.weight_decay,
        'num_workers':        args.num_workers,
        'fine_tune':          args.fine_tune,
        'unfreeze_encoder_layer4': args.unfreeze_encoder_layer4
    }

    train_model(
        config=cfg,
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        resume_model=args.resume_model
    )
