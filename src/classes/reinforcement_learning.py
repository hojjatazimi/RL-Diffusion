import torch
from tqdm import tqdm
import logging
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    nb_epochs: int = 150_000
    batch_size: int = 6_000
    t_initial: int = 40
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    min_reward_improvement: float = 1e-4
    checkpoint_frequency: int = 1000
    log_frequency: int = 100
    update_frequency: int = 32  # Number of trajectories to collect before updating
    gradient_accumulation_steps: int = 4  # Number of backward passes before optimizer step

class BatchReinforcementLearningTrainer:
    def __init__(
        self, 
        model: torch.nn.Module,
        reward_function: Callable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Optional[TrainingConfig] = None
    ):
        """Initialize the Batch RL Trainer."""
        self.model = model
        self.reward_function = reward_function
        self.optimizer = optimizer
        self.device = device
        self.config = config or TrainingConfig()
        self.logger = self._setup_logger()
        
        # Training state
        self.best_reward = float('-inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Initialize replay buffer for batch updates
        self.trajectory_buffer = {
            'states': [],
            'log_probs': [],
            'rewards': []
        }

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for training monitoring."""
        logger = logging.getLogger('BatchRLTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def clear_trajectory_buffer(self):
        """Clear the trajectory buffer after an update."""
        self.trajectory_buffer = {
            'states': [],
            'log_probs': [],
            'rewards': []
        }

    def add_to_buffer(self, state, log_probs, reward):
        """Add a trajectory to the buffer."""
        self.trajectory_buffer['states'].append(state)
        self.trajectory_buffer['log_probs'].append(log_probs)
        self.trajectory_buffer['rewards'].append(reward)

    def process_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process collected trajectories into training batch."""
        # Stack all trajectories
        states = torch.stack(self.trajectory_buffer['states'])
        log_probs = torch.stack([torch.stack(lp) for lp in self.trajectory_buffer['log_probs']])
        rewards = torch.stack(self.trajectory_buffer['rewards'])

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        return states, log_probs, rewards

    def compute_batch_loss(self, log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch of trajectories."""
        # Compute policy gradient loss for all trajectories
        policy_loss = -rewards.unsqueeze(1) * log_probs.sum(dim=1)
        return policy_loss.mean()

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch using batch updates."""
        total_loss = 0
        total_reward = 0
        num_updates = 0
        
        # Clear gradient accumulation at the start of epoch
        self.optimizer.zero_grad()
        
        for batch_idx in range(self.config.update_frequency):
            # Sample new batch
            x0 = torch.from_numpy(self.sample_batch(self.config.batch_size)).float().to(self.device)
            
            # Forward process
            with torch.no_grad():
                _, _, x = self.model.forward_process(x0, self.config.t_initial)
            
            # Collect trajectory
            current_log_probs = []
            for t in range(self.config.t_initial, 0, -1):
                x, log_prob, _, _ = self.model.select_action(self.model, x, t)
                current_log_probs.append(log_prob)
            
            # Calculate reward
            reward = self.reward_function(x)
            if torch.isnan(reward).any():
                raise ValueError(f"NaN detected in reward at epoch {epoch}, batch {batch_idx}")
            
            # Add to buffer
            self.add_to_buffer(x, current_log_probs, reward)
            total_reward += reward.mean().item()
            
            # If buffer is full, perform update
            if len(self.trajectory_buffer['states']) >= self.config.update_frequency:
                states, log_probs, rewards = self.process_batch()
                loss = self.compute_batch_loss(log_probs, rewards)
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.max_grad_norm
                    )
                    
                    # Perform optimization step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                num_updates += 1
                
                # Clear buffer after update
                self.clear_trajectory_buffer()
        
        avg_loss = total_loss / num_updates if num_updates > 0 else float('inf')
        avg_reward = total_reward / self.config.update_frequency
        
        return avg_loss, avg_reward

    def train(self) -> Tuple[List[float], List[float]]:
        """Train the model with batch updates."""
        training_loss = []
        rewards = []
        
        try:
            for epoch in tqdm(range(self.config.nb_epochs)):
                loss, reward = self.train_epoch(epoch)
                training_loss.append(loss)
                rewards.append(reward)
                
                # Logging and checkpointing logic remains the same
                if epoch % self.config.log_frequency == 0:
                    self.logger.info(
                        f"Epoch {epoch}: Loss = {loss:.4f}, Reward = {reward:.4f}"
                    )
                
                if epoch % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint(
                        epoch, 
                        {'loss': loss, 'reward': reward},
                        f'checkpoint_epoch_{epoch}.pt'
                    )
                
                if self.check_early_stopping(reward):
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    self.model.load_state_dict(self.best_model_state)
                    break
                    
        except Exception as e:
            self.logger.error(f"Training interrupted: {str(e)}")
            self.save_checkpoint(
                epoch, 
                {'training_loss': training_loss, 'rewards': rewards},
                'emergency_checkpoint.pt'
            )
            raise
            
        return training_loss, rewards