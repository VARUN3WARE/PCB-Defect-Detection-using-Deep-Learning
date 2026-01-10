"""
PCB Defect Detection - Model Training Script
Trains ResNet-18 based classifier with regularization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt


# Defect class mapping
DEFECT_CLASSES = {
    1: 'open',
    2: 'short',
    3: 'mousebite',
    4: 'spur',
    5: 'copper',
    6: 'pin-hole'
}


class PCBDataset(Dataset):
    """PyTorch Dataset for PCB defect detection"""
    
    def __init__(self, image_paths, annotations, transform=None):
        """
        Args:
            image_paths: List of image file paths
            annotations: List of defect labels (multi-label binary vectors)
            transform: Optional transform to apply to images
        """
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.FloatTensor(self.annotations[idx])
        
        return image, label


class PCBDefectClassifier(nn.Module):
    """ResNet-18 based multi-label classifier with regularization"""
    
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(PCBDefectClassifier, self).__init__()
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace final FC layer with dropout for regularization
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.resnet(x)


def parse_annotation(anno_file):
    """Parse annotation file with space-separated format"""
    defects = []
    if not anno_file.exists():
        return defects
    
    try:
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split()  # Space-separated, not comma!
                if len(parts) >= 5:
                    defects.append({
                        'x1': int(parts[0]),
                        'y1': int(parts[1]),
                        'x2': int(parts[2]),
                        'y2': int(parts[3]),
                        'type': int(parts[4])
                    })
    except Exception as e:
        print(f"Warning: Error parsing {anno_file}: {e}")
    return defects


def load_dataset(dataset_path, split='train'):
    """
    Load PCB dataset from DeepPCB format
    
    Args:
        dataset_path: Path to DeepPCB dataset
        split: 'train', 'val', or 'test'
        
    Returns:
        image_paths: List of image paths
        labels: List of multi-label vectors
    """
    dataset_path = Path(dataset_path)
    
    # Load test.txt (we'll split it into train/val/test)
    test_txt = dataset_path / 'PCBData' / 'test.txt'
    
    # Check if dataset exists
    if not test_txt.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Please download the DeepPCB dataset:\n"
            f"  git clone https://github.com/tangsanli5201/DeepPCB.git ../DeepPCB\n"
            f"Or specify the correct path with --dataset flag"
        )
    
    with open(test_txt, 'r') as f:
        test_entries = f.read().strip().split('\n')
    
    # Split dataset: 70% train, 15% val, 15% test
    np.random.seed(42)
    indices = np.random.permutation(len(test_entries))
    
    train_size = int(0.7 * len(test_entries))
    val_size = int(0.15 * len(test_entries))
    
    if split == 'train':
        selected_indices = indices[:train_size]
    elif split == 'val':
        selected_indices = indices[train_size:train_size + val_size]
    else:  # test
        selected_indices = indices[train_size + val_size:]
    
    image_paths = []
    labels = []
    
    for idx in selected_indices:
        entry = test_entries[idx]
        parts = entry.split()
        img_rel_path = parts[0]
        anno_rel_path = parts[1]
        
        # Construct paths
        image_id = Path(img_rel_path).stem
        img_dir = Path(img_rel_path).parent
        
        test_img_path = dataset_path / 'PCBData' / img_dir / f"{image_id}_test.jpg"
        anno_path = dataset_path / 'PCBData' / anno_rel_path
        
        if not test_img_path.exists():
            continue
        
        # Parse annotations
        defects = parse_annotation(anno_path)
        
        # Create multi-label vector (6 classes)
        label_vector = [0] * 6
        for defect in defects:
            defect_type = defect['type']
            if 1 <= defect_type <= 6:
                label_vector[defect_type - 1] = 1
        
        image_paths.append(str(test_img_path))
        labels.append(label_vector)
    
    return image_paths, labels


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate classification metrics"""
    pred_binary = (predictions > threshold).astype(int)
    target_binary = targets.astype(int)
    
    # Flatten for overall metrics
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    
    # Calculate metrics (use 'binary' for flattened arrays)
    try:
        f1 = f1_score(target_flat, pred_flat, average='binary', zero_division=0)
        precision = precision_score(target_flat, pred_flat, average='binary', zero_division=0)
        recall = recall_score(target_flat, pred_flat, average='binary', zero_division=0)
    except:
        # Fallback to samples if binary fails
        f1 = f1_score(target_binary, pred_binary, average='samples', zero_division=0)
        precision = precision_score(target_binary, pred_binary, average='samples', zero_division=0)
        recall = recall_score(target_binary, pred_binary, average='samples', zero_division=0)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    for images, labels in tqdm(dataloader, desc='Training', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_predictions.append(outputs.detach().cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, metrics


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss, metrics, all_predictions


def plot_training_history(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 score plot
    axes[1].plot(history['train_f1'], label='Train F1', marker='o')
    axes[1].plot(history['val_f1'], label='Val F1', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Training and Validation F1 Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_model(config):
    """
    Main training function
    
    Args:
        config: Dictionary with training configuration
    """
    print("="*70)
    print("PCB DEFECT DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load datasets
    print(f"\n📂 Loading dataset from: {config['dataset_path']}")
    train_images, train_labels = load_dataset(config['dataset_path'], 'train')
    val_images, val_labels = load_dataset(config['dataset_path'], 'val')
    test_images, test_labels = load_dataset(config['dataset_path'], 'test')
    
    print(f"   Train samples: {len(train_images)}")
    print(f"   Val samples: {len(val_images)}")
    print(f"   Test samples: {len(test_images)}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PCBDataset(train_images, train_labels, train_transform)
    val_dataset = PCBDataset(val_images, val_labels, val_transform)
    test_dataset = PCBDataset(test_images, test_labels, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, num_workers=2)
    
    # Initialize model
    print(f"\n🧠 Initializing model...")
    model = PCBDefectClassifier(num_classes=6, dropout_rate=config['dropout_rate'])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    print("="*70)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-"*70)
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train F1: {train_metrics['f1']*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val F1:   {val_metrics['f1']*100:.2f}%")
        print(f"Val Precision: {val_metrics['precision']*100:.2f}% | Val Recall: {val_metrics['recall']*100:.2f}%")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            model_save_path = Path(config['output_dir']) / 'best_model_regularized.pth'
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved! (F1: {best_val_f1*100:.2f}%)")
        
        # Save checkpoint every epoch as backup
        if epoch == 0 or (epoch + 1) % 5 == 0:
            checkpoint_path = Path(config['output_dir']) / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
    
    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"   Best Val F1: {best_val_f1*100:.2f}% at epoch {best_epoch}")
    print("="*70)
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    
    # Load best model if it exists, otherwise use current model
    best_model_path = Path(config['output_dir']) / 'best_model_regularized.pth'
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"   Loaded best model from epoch {best_epoch}")
    else:
        print(f"   Using model from last epoch (no best model saved)")
    
    test_loss, test_metrics, test_predictions = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    
    # Save training history
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    plot_path = output_dir / 'training_history.png'
    plot_training_history(history, plot_path)
    print(f"\n📈 Training curves saved to: {plot_path}")
    
    # Save metrics
    results = {
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'test_metrics': {
            'loss': float(test_loss),
            'f1': float(test_metrics['f1']),
            'precision': float(test_metrics['precision']),
            'recall': float(test_metrics['recall'])
        },
        'config': config,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved to: {results_path}")
    
    return model, history, test_metrics


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PCB Defect Detection Model')
    parser.add_argument('--dataset', type=str, default='../DeepPCB',
                       help='Path to DeepPCB dataset')
    parser.add_argument('--output', type=str, default='../models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    
    args = parser.parse_args()
    
    config = {
        'dataset_path': args.dataset,
        'output_dir': args.output,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'dropout_rate': args.dropout,
        'weight_decay': args.weight_decay,
        'use_cuda': not args.no_cuda
    }
    
    # Train model
    train_model(config)


if __name__ == '__main__':
    main()
