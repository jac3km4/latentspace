use ndarray::{Array1, Array4, ArrayView4};
use rand::Rng;

use super::{Scheduler, Timesteps};

#[derive(Debug)]
pub struct Euler {
    alphas_cumprod: Array1<f32>,
    init_sigma: f32,
    num_steps: usize,
}

impl Euler {
    pub fn new(num_train_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        // let betas = Array1::linspace(beta_start, beta_end, num_train_timesteps);
        let mut betas = Array1::linspace(beta_start.sqrt(), beta_end.sqrt(), num_train_timesteps);
        betas.iter_mut().for_each(|f| *f = f.powi(2));

        let alphas = 1.0 - betas;
        let alphas_cumprod: Array1<_> = alphas
            .iter()
            .scan(1.0, |state, &x| {
                *state *= x;
                Some(*state)
            })
            .collect();
        let init_sigma = alphas_cumprod
            .iter()
            .map(|&x| ((1. - x) / x).sqrt())
            .max_by_key(|x| x.to_bits())
            .unwrap_or(0.);

        Self {
            alphas_cumprod,
            init_sigma,
            num_steps: num_train_timesteps,
        }
    }
}

impl Scheduler for Euler {
    fn step(
        &mut self,
        index: usize,
        steps: &Timesteps,
        latent: ArrayView4<'_, f32>,
        output: ArrayView4<'_, f32>,
        _rng: &mut impl Rng,
    ) -> Array4<f32> {
        &latent + &output * (steps.sigmas[index + 1] - steps.sigmas[index])
    }

    fn timesteps(&mut self, num_inference_steps: usize) -> Timesteps {
        let timesteps = Array1::linspace(self.num_steps as f32 - 1., 0., num_inference_steps);

        let sigmas_range: Vec<_> = (0..self.alphas_cumprod.len()).map(|x| x as f32).collect();
        let sigmas: Vec<_> = self
            .alphas_cumprod
            .iter()
            .map(|&x| ((1. - x) / x).sqrt())
            .collect();
        let sigmas = timesteps
            .iter()
            .map(|&x| lin_interpolate(&sigmas_range, &sigmas, x))
            .chain([0.])
            .collect();

        Timesteps { sigmas, timesteps }
    }

    fn sigma_multiplier(&self) -> f32 {
        self.init_sigma
    }

    fn scale_multiplier(&self, sigma: f32) -> f32 {
        1. / (sigma.powi(2) + 1.).sqrt()
    }
}

impl Default for Euler {
    fn default() -> Self {
        Self::new(1000, 0.00085, 0.012)
    }
}

fn lin_interpolate(lhs: &[f32], rhs: &[f32], x: f32) -> f32 {
    let i = lhs
        .binary_search_by_key(&x.to_bits(), |f| f.to_bits())
        .unwrap_or_else(|x| x)
        .min(lhs.len() - 2);

    let x1 = lhs[i];
    let x2 = lhs[i + 1];
    let y1 = rhs[i];
    let y2 = rhs[i + 1];
    y1 + (y2 - y1) * (x2 - x1) / (x2 - x1)
}
