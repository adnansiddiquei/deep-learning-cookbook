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

So this means that you can have different times of "noising schedules". In the example below we use
a linear noise schedule.
This simply means that beta_t ranges linearly between [beta_1, beta_2].

So for example, if beta_t ranges linearly [1e-4, 0.2] with total timesteps T = 1000 (as chosen by
Ho, J., et al (2020)), then
    At t = 100      alpha_t = 0.3656, sqrt(alpha_t) = 0.6046
                    which means that 60% of the data is signal, the rest is noise

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
"""

import torch
import torch.nn as nn

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)


class DDPM(nn.Module):
    def __init__(self, decoder: nn.Module):
        super().__init__()

        self.decoder = decoder

    def _compute_noise_schedule(
        self, beta_1: float, beta_2: float, T: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes a linear noise schedule.

        Args:
            beta_1: Lower bound for noise schedule.
            beta_2: Upper bound for noise schedule.
            T: Total number of timesteps in the forward and reverse process.
        """
        assert 0.0 < beta_1 < beta_2 <= 1.0, 'beta1 and beta2 must be in [0, 1]'

        # Create a linear schedule from `beta1` to `beta2` over `T` timesteps
        steps = torch.arange(0, T + 1, dtype=torch.float32)
        beta_t = (beta_2 - beta_1) * (steps / T) + beta_1

        # Compute the corresponding `alpha_t` schedule (Sec 18.2.1, Eqn, 18.7; Prince)
        # Cumprod in log-space (better precision) that simple prod
        alpha_t = torch.exp(torch.cumsum(torch.log(1 - beta_t), dim=0))

        return beta_t, alpha_t
