import os
import sys

# Determine base project path
DRIVE_BASE = '/content/drive/MyDrive/cam-vessel-ai'
if os.path.isdir(DRIVE_BASE):
    # Running in Colab with Drive available
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
    except Exception:
        pass
    os.chdir(DRIVE_BASE)
    base_path = DRIVE_BASE
    print(f"Working directory set to: {base_path}")
else:
    # Fallback for local or other environments
    base_path = os.getcwd()
    print(f"Base project path: {base_path}")

# Ensure project root is in sys.path
sys.path.append(base_path)

from train import train_model

if __name__ == '__main__':
    # Config for training
    config = {
        'encoder': 'resnet34',
        'pretrained': True,
        'epochs': 20,
        'batch_size': 32,
        'lr': 2e-6,
        'fine_tune': True,
        'unfreeze_encoder_layer4': True
    }

    # Paths
    dataset_path = os.path.join(os.getcwd(), 'data', 'retina')
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints')

    # Resolve latest checkpoint (epoch_*.pth only)
    checkpoint_dir = os.path.join(checkpoint_path, config['encoder'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpts = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') and f.endswith('.pth')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    resume_model = os.path.join(checkpoint_dir, ckpts[-1]) if ckpts else None

    print(f"Starting training with config: {config}")
    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    if resume_model:
        print(f"Resuming from: {resume_model}")

    train_model(
        config=config,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        resume_model=resume_model
    )
