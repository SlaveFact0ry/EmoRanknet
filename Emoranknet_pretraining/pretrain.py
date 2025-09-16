import os
import argparse
from dataclasses import dataclass, InitVar

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Average
from ignite.handlers import ModelCheckpoint, ProgressBar


def load_x3d_model():
    """
    Loads pre-trained X3D model.
    """
    print("Loading pre-trained X3D model from torch.hub...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_m', pretrained=True)
    model = model.eval()
    model = model.to("cuda")
    model = model.drop_layer(model.blocks[-1])
    return model

def preprocess_video_for_x3d(video_path):
    """
    Loads and preprocesses video files into a tensor.
    """
    return torch.randn(3, 16, 224, 224) # (C, T, H, W)


class IntensityScorer(nn.Module):
    """
    A wrapper around the X3D model that freezes the base and trains a
    linear head to output a single intensity score.
    """
    def __init__(self, freeze_base=True):
        super().__init__()
        base_model = load_x3d_model()
        
        self.feature_extractor = nn.Sequential(*list(base_model.blocks.children()))
        
        feature_dim = base_model.blocks[5].proj.in_features #2048

        self.scoring_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

        if freeze_base:
            print("Freezing base feature extractor layers.")
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, T, H, W)
        features = self.feature_extractor(x)
        # The output of X3D blocks is often pooled, resulting in (B, embedding_dim)
        score = self.scoring_head(features)
        return score.squeeze(-1) # Return shape (B,)

#  Piawised dataset

class PairwiseVideoDataset(Dataset):
    """
    A dataset that yields pairs of videos and a target indicating
    which one has a higher intensity score.
    """
    def __init__(self, df: pd.DataFrame, video_root: str):
        self.df = df.reset_index(drop=True)
        self.video_root = video_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        item1 = self.df.iloc[idx]

        while (idx2 := np.random.randint(len(self.df))) == idx:
            pass
        item2 = self.df.iloc[idx2]

        score1 = item1['intensity_score']
        score2 = item2['intensity_score']
        
        if score1 > score2:
            target = 1.0
        elif score2 > score1:
            target = 0.0
        else: 
            target = 0.5
        
        video_path1 = os.path.join(self.video_root, item1['video_path'])
        video_path2 = os.path.join(self.video_root, item2['video_path'])

        video_tensor1 = preprocess_video_for_x3d(video_path1)
        video_tensor2 = preprocess_video_for_x3d(video_path2)
        
        return video_tensor1, video_tensor2, torch.tensor(target, dtype=torch.float32)

#  Ranknet loss and Ignite engines

def ranknet_loss(s1: Tensor, s2: Tensor, t: Tensor) -> Tensor:
    """The RankNet loss function."""
    o = torch.sigmoid(s1 - s2)
    loss = (-t * o + F.softplus(o)).mean()
    return loss

def prepare_batch(batch: tuple, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Helper to move a batch to the correct device."""
    return tuple(t.to(device) for t in batch)

@dataclass
class RankNetTrainer(Engine):
    """Ignite engine for handling the training update step."""
    net: InitVar[nn.Module]
    opt: Optimizer
    device: torch.device
    
    def __post_init__(self, net: nn.Module):
        super().__init__(self._update)
        self._net = net

    def _update(self, engine: Engine, batch: tuple) -> Tensor:
        x1, x2, t = prepare_batch(batch, self.device)
        self._net.train()
        self.opt.zero_grad()
        s1, s2 = self._net(x1), self._net(x2)
        loss = ranknet_loss(s1, s2, t)
        loss.backward()
        self.opt.step()
        return loss.item()

@dataclass
class RankNetEvaluator(Engine):
    """Ignite engine for handling the evaluation/validation step."""
    net: InitVar[nn.Module]
    device: torch.device
    
    def __post_init__(self, net: nn.Module):
        super().__init__(self._inference)
        self._net = net
        
        def _acc_output_transform(output: tuple) -> tuple[Tensor, Tensor]:
            s1, s2, t, _ = output
            # Prediction is 1 if s1 > s2, else 0
            # Target is 1 if t > 0.5, else 0
            return (s1 > s2).long(), (t > 0.5).long()

        Average(output_transform=lambda out: out[3]).attach(self, "loss")
        Accuracy(output_transform=_acc_output_transform).attach(self, "accuracy")

    @torch.no_grad()
    def _inference(self, engine: Engine, batch: tuple):
        x1, x2, t = prepare_batch(batch, self.device)
        self._net.eval()
        s1, s2 = self._net(x1), self._net(x2)
        loss = ranknet_loss(s1, s2, t)
        return s1, s2, t, loss.item()

#  Main training script

def main(args):
    """Main function to run the pre-training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #  Data Loading
    print(f"Loading data from {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    
    train_dataset = PairwiseVideoDataset(train_df, args.video_root)
    val_dataset = PairwiseVideoDataset(val_df, args.video_root)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Data loaded: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

    #  Model and Optimizer
    model = IntensityScorer().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    #  Trainer and Evaluator Setup
    trainer = RankNetTrainer(model, optimizer, device)
    evaluator = RankNetEvaluator(model, device)
    
    # Add progress bars
    ProgressBar().attach(trainer, output_transform=lambda loss: {'train_loss': loss})
    ProgressBar().attach(evaluator)
    
    #  Event Handlers
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine: Engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch {engine.state.epoch} - Val Loss: {metrics['loss']:.4f}, Val Accuracy: {metrics['accuracy']:.4f}")

    # Checkpoint
    checkpointer = ModelCheckpoint(
        dirname=args.output_dir,
        filename_prefix="best",
        n_saved=1,
        create_dir=True,
        require_empty=False,
        score_name="accuracy",
        score_function=lambda engine: engine.state.metrics['accuracy'],
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpointer, {'model': model})

    #  Run Training
    print("\n--- Starting Pre-training ---")
    trainer.run(train_loader, max_epochs=args.epochs)
    print("--- Pre-training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train an X3D intensity scorer with RankNet loss.")
    parser.add_argument("--data_csv", type=str, required=True, help="Path to the CSV file with video_path and intensity_score.")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory where video files are stored.")
    parser.add_argument("--output_dir", type=str, default="./Emoranknet_pretraining_output", help="Directory to save model checkpoints.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_tument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    
    args = parser.parse_args()
    main(args)
