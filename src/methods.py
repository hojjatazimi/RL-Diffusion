import torch
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Callable
import logging
from sklearn.datasets import make_swiss_roll
import os
from classes.policy_network import PolicyNetwork
from classes.diffusion_model import DiffusionModel

def train_rl(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    reward_function: Callable,
    weights,
    nb_epochs: int = 150_000,
    batch_size: int = 6_000,
    device: str = 'cpu',
    max_grad_norm: float = 1.0,
    early_stop_patience: int = 5
) -> Tuple[List[float], List[float]]:
    """
    Train a model using reinforcement learning.
    
    Args:
        model: The model to train
        optimizer: The optimizer to use
        reward_function: Function that computes rewards
        nb_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to run training on ('cpu' or 'cuda')
        max_grad_norm: Maximum gradient norm for clipping
        early_stop_patience: Number of epochs to wait before early stopping
    
    Returns:
        Tuple of (training_losses, rewards)
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize tracking variables
    training_loss = []
    rewards = []
    best_reward = float('-inf')
    patience_counter = 0
    
    try:
        for epoch in tqdm(range(nb_epochs)):
            # Forward pass
            x0 = torch.from_numpy(sample_batch(batch_size)).float().to(device)
            t = 40
            mu_posterior, sigma_posterior, x = model.forward_process(x0, t)
            
            # Collect actions and log probabilities
            log_probs = []
            for t in range(40, 0, -1):
                x, log_prob, _, _ = model.select_action(x, t)
                log_probs.append(log_prob)
            
            # Compute reward and check for NaN
            reward = reward_function(x, weights)
            if isinstance(reward, (float, int)):
                reward = torch.tensor([reward], dtype=torch.float32).to(device)
            else:
                reward = torch.tensor(reward, dtype=torch.float32).to(device)
                
            if torch.isnan(reward).any():
                logger.error(f"NaN detected in reward at epoch {epoch}")
                break
            
            rewards.append(reward.item() if reward.dim() == 0 else reward.mean().item())
            
            # Compute loss
            log_probs_stack = torch.stack(log_probs)
            log_probs_sum = log_probs_stack.sum()
            loss = -reward * log_probs_sum
            
            if torch.isnan(loss).any():
                logger.error(f"NaN detected in loss at epoch {epoch}")
                break
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Check gradients for NaN
            if any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None):
                logger.error(f"NaN detected in gradients at epoch {epoch}")
                break
            
            optimizer.step()
            training_loss.append(loss.item())
            
            # Check parameters for NaN
            if any(torch.isnan(param).any() for param in model.parameters()):
                logger.error(f"NaN detected in parameters at epoch {epoch}")
                break
            
            # Early stopping check
            current_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else rewards[-1]
            if current_reward > best_reward:
                best_reward = current_reward
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Periodic logging
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}, Reward = {rewards[-1]:.4f}")
                
    except Exception as e:
        logger.error(f"Training interrupted due to error: {str(e)}")
        raise
        
    return training_loss, rewards

def sample_batch(size):
    x, _ = make_swiss_roll(size)
    x = x[:, [2, 0]] / 10.0 * np.array([1, -1])
    return x[:, 0].reshape((1, size))

def reward_function(features, weights, bias=None):
    # Calculate the linear combination
    logits = torch.matmul(features, weights)

    # If bias is provided, add it to the logits
    if bias is not None:
        logits += bias

    # Apply the sigmoid function to get the probabilities
    probabilities = torch.sigmoid(logits)

    return probabilities

def list_files_in_directory(directory):
    """
    List all files in the given directory, including their full paths.

    :param directory: Path to the directory
    :return: List of full file paths
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def init_model(device, state_size, action_size):
    policy_net = PolicyNetwork(state_size, action_size, device=device).to(device)
    model = DiffusionModel(policy_net, device=device)
    return model

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    epoch = checkpoint['epoch']
    return model, epoch