"""
This file implements the Diffusion denoising probabilistic model (DDPM) as originaly defined by
Ho, J., et al (2020).

The concepts explored here are explained in more detail in Ch. 18 of Understanding Deep Learning
by Simon Prince.

A DDPM consists of an encoder and a decoder.
 - The encoder a data sample `x` maps it through a series of steps to standard normal noise.
 - The decoder learns to reverse this process to generate realistic samples of `x`.

The Encoder (forward process)
-----------------------------
Over a series of timesteps (e.g., 1000 steps), the encoder adds Gaussian noise to a given data
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

The Decoder (reverse process)
-----------------------------
This can be any model, but the objective of this model is to learn the noise (eps) added to the
latent at any given step, to get to the next step.

The noise added to the latents at any given step is always scaled standard normal noise, so
effectively the decoder needs to learn to predict this noise accurately in order to denoise the
latent variables and reconstruct the original data sample `x`.

So the decoder simply needs to predict standard normal noise at each step.


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

    def _compute_noise_schedule(self, beta1, beta2, num_timesteps):
        """

        Noise added

        Args:
            beta1 (_type_): _description_
            beta2 (_type_): _description_
            num_timesteps (_type_): _description_
        """
        assert beta1 < beta2 < 1.0, 'beta1 and beta2 must be in (0, 1)'
