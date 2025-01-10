import torch
import torch.nn as nn
from typing import Tuple, Optional


class DiffusionModel(nn.Module):
    def __init__(self, model: nn.Module, n_steps: int = 40, device: str = 'cpu'):
        """
        Diffusion Model for forward and reverse processes.

        Args:
            model (nn.Module): Neural network model for parameterizing the reverse process.
            n_steps (int): Number of diffusion steps.
            device (str): Device to run computations on.
        """
        super().__init__()
        self.policy = model
        self.device = device
        self.n_steps = n_steps

        # Compute beta schedule
        betas = torch.linspace(-18, 10, n_steps)
        self.beta = torch.sigmoid(betas) * (3e-1 - 1e-5) + 1e-5

        # Compute alpha and cumulative alpha schedules
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

    def forward_process(self, x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process.

        Args:
            x0 (torch.Tensor): Original data.
            t (int): Current time step.

        Returns:
            Tuple containing the mean (mu), standard deviation (sigma), and noisy sample (xt).
        """
        t = t - 1  # Convert to zero-based indexing
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]

        # Sample noisy data xt
        noise = torch.randn_like(x0)
        xt = x0 * torch.sqrt(alpha_bar_t) + noise * torch.sqrt(1.0 - alpha_bar_t)

        # Compute forward process parameters
        mu1_scale = torch.sqrt(alpha_bar_t / alpha_t)
        mu2_scale = 1.0 / torch.sqrt(alpha_t)
        cov1 = 1.0 - alpha_bar_t / alpha_t
        cov2 = beta_t / alpha_t
        lam = 1.0 / cov1 + 1.0 / cov2

        mu = (x0 * mu1_scale / cov1 + xt * mu2_scale / cov2) / lam
        sigma = torch.sqrt(1.0 / lam)

        return mu, sigma, xt

    def reverse(self, xt: torch.Tensor, t: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Reverse diffusion process.

        Args:
            xt (torch.Tensor): Noisy data at time step t.
            t (int): Current time step.

        Returns:
            Tuple containing the predicted mean (mu), standard deviation (sigma), and denoised sample.
        """
        t = t - 1  # Convert to zero-based indexing
        if t == 0:
            # Final step, no further noise addition
            return None, None, xt

        # Predict mean and log-variance from the model
        mu_logvar = self.policy(xt, t)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        sigma = torch.sqrt(torch.exp(logvar))

        # Sample new data point
        noise = torch.randn_like(xt)
        samples = mu + noise * sigma

        # Sanity check for NaN values
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            raise ValueError(f"NaN detected at step {t}: mu or sigma is NaN.\nmu: {mu}\nsigma: {sigma}")

        return mu, sigma, samples

    def sample(self, size: int, device: Optional[str] = None) -> torch.Tensor:
        """
        Generate samples from the reverse process.

        Args:
            size (int): Number of samples to generate.
            device (str, optional): Device for computations. Defaults to model's device.

        Returns:
            torch.Tensor: Generated samples.
        """
        device = device or self.device
        noise = torch.randn((size, 2), device=device)
        samples = [noise]

        for step in range(self.n_steps, 0, -1):
            _, _, xt = self.reverse(samples[-1], step)
            samples.append(xt)

        return samples[-1]  # Return the final denoised sample