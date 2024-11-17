"""
This file implements the Diffusion denoising probabilistic model (DDPM) as originaly defined by
Denoising Diffusion Probabilistic Models Ho, J., et al (2020).

The concepts explored here are explained in more detail in Ch. 18 of Understanding Deep Learning
by Simon Prince.

A DDPM consists of an encoder and a decoder.
 - The encoder a data sample `x` maps it through a series of steps to standard normal noise.
 - The decoder learns to reverse this process to generate realistic samples of `x`.

The Encoder (forward process)
-----------------------------
Over a series of timesteps T (e.g., 1000 steps), the encoder adds Gaussian noise to a given data
sample `x` (e.g., this could be a 3x256x256 RGB image) as well as attenuating the original signal.

In this case, over 1000 steps, the encoder maps `x` through latent variables z_1, z_2, ..., z_1000.

Take the timesteps t ∈ [0, 1000].
    At t = 0,       z_0 = x
    At t = 1,       z_1 = sqrt(1 - beta_1) * x + sqrt(beta_1) * eps_1
    At t = 30,      z_30 = sqrt(1 - beta_30) * z_29 + sqrt(beta_30) * eps_30

Where eps_1, ..., eps_1000 are simply samples drawn from a standard normal distribution.
Note the first term attenuates the previous latent, and the secong term adds noise.

The above process is termed a Markov chain because any given latent variable only depends on the
state of the previous latent variable.
After enough steps, the data sample `x` is completely noise.

We can write the above in closed form (Ch. 18.2.1. UDL, Prince):
    At t = t,       z_t = sqrt(alpha_t) * x + sqrt(1 - alpha_t) * eps
    where           alpha_t = prod(1 - beta_i) for i in range(1, t+1)
    and             eps is standard normal noise.

So this means that you can have different times of "noising schedules".
In the example below we use a linear noise schedule.
This simply means that beta_t ranges linearly between [beta_1, beta_2].

So for example, if beta_t ranges linearly [1e-4, 0.2] with total timesteps T = 1000 (as chosen by
Ho, J., et al (2020)), then
    At t = 100      alpha_t = 0.3656, sqrt(alpha_t) = 0.6046

The Decoder (reverse process)
-----------------------------
The decoder's objective is to learn the noise (eps) added to the latent variable at any given step,
in order to reverse the process and reconstruct the original data sample x.

During the forward process, noise is added to the data sample over a series of timesteps, resulting
in a noisy latent variable. The decoder's task is to predict this noise accurately at each step,
allowing it to denoise the latent variables and reconstruct the original data sample.

The decoder receives a latent variable (composed of an attenuated signal and noise) and the current
timestep as inputs. It must predict the specific noise that was added to the latent variable at the
previous step, enabling the reconstruction of the original data sample.

Training and sampling
---------------------
See Algorithm 18.1 and 18.2 from Prince (2023).

These algorithms detail how the forward training process, and the backward sampling process in the
below model works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_noise_schedule(
    beta_1: float = 1e-4,
    beta_2: float = 0.2,
    T: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes a linear noise (variance) schedule.

    The default paramters are exactly as used by Ho, J., et al (2020) in their paper.

    Args:
        beta_1: The variance of the Gaussian noise added by the encoder in the first encoder
                step.
        beta_2: The variance of the Gaussian noise add in the last encoder step.
        T: Total number of timesteps in the forward (encoder) process.
    """
    assert 0.0 < beta_1 < beta_2 <= 1.0, 'beta1 and beta2 must be in [0, 1]'

    # Create a linear schedule from `beta1` to `beta2` over `T` timesteps
    steps = torch.arange(0, T + 1, dtype=torch.float32)
    beta_t = (beta_2 - beta_1) * (steps / T) + beta_1

    # Compute the corresponding `alpha_t` schedule (Sec 18.2.1, Eqn, 18.7; Prince)
    # Cumprod in log-space provides better precision than simple product
    alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))

    return beta_t, alpha_t


class DDPM(nn.Module):
    def __init__(self, decoder: nn.Module, beta_t: torch.Tensor, alpha_t: torch.Tensor):
        super().__init__()

        # This is the decoder model, that will learn the backward process
        self.decoder = decoder

        # Register as a buffer so that it is not updated by the optimizer
        self.register_buffer('beta_t', beta_t)
        self.beta_t
        self.register_buffer('alpha_t', alpha_t)
        self.alpha_t
        assert len(self.beta_t) == len(self.beta_t)

        # The number of diffusion steps is the number of elements in the noise schedule
        self.n_T = len(self.beta_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model. Algorithm 18.1 in Prince (2023).

        Args:
            x: A data sample. Should be of shape
                (batch_size, channels, height, width) or
                (batch_size, channels, length)
        """
        assert (
            len(x.shape) in [3, 4]
        ), 'Input should be of shape (batch_size, channels, height, width) or (batch_size, channels, length)'

        batch_size = x.shape[0]
        num_data_dimensions = len(x.shape[1:])  # the number of dimensions in the data

        # Sample a random timestep for each element item in the batch
        t = torch.randint(1, self.n_T, (x.shape[0],), device=x.device)

        # Sample standard normal noise, with the same shape as x
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        # Get the alpha_t value for each of the corresponding timesteps
        alpha_t = self.alpha_t[t]

        # Reshape alpha_t so it is ready for broadcasting
        alpha_t = alpha_t.view(
            batch_size,
            *[1 for _ in range(num_data_dimensions)],
            # ^ will be either (batch_size, 1, 1, 1) or (batch_size, 1, 1)
        )

        # Add noise to each sample. See notes above.
        # This is the closed form expression for the noise added at step t.
        z_t = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps
        # z_t.shape == x.shape

        # Now we get the model to predict the noise added at the give timestep
        preds = self.decoder(z_t, t / self.n_T)

        # Compute the loss
        return F.mse_loss(preds, eps)

    def sample(self, shape: list, device):
        """Reverse diffusion (decoder) process. Generate samples using Algorithm 18.2 in Prince
        (2023).

        Args:
            shape: The shape of the noise tensor to begin the reverse diffusion with. E.g., if you
                    want to generate 100 samples, of 3 channel 28x28 images, then this should be
                    (100, 3, 28, 28).
            device: 'cpu', 'cuda' or 'mps'.
        """
        z_t = torch.randn(shape, device=device)
        _ones = torch.ones(
            shape[0], device=device
        )  # a tensor of ones, for reshaping (t / self.n_T)

        def backward_step(z_t, t):
            alpha_t = self.alpha_t[t]  # Get the alpha_t value for the current time step
            beta_t = self.beta_t[t]  # Get the beta_t value for the current time step

            # Get the noise predictions for the current step
            noise_preds = self.decoder(z_t, (t / self.n_T) * _ones)

            # Predit the previous latent variable
            z_t -= (beta_t / torch.sqrt(1 - alpha_t)) * noise_preds
            z_t /= torch.sqrt(1 - beta_t)

            return z_t

        for t in range(self.n_T, 1, -1):
            z_t = backward_step(z_t=z_t, t=t)

            # Add some more noise
            beta_t = self.beta_t[t]
            z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)

        # Do 1 more step, without adding any noise this time
        z_t = backward_step(z_t=z_t, t=0)

        return z_t
