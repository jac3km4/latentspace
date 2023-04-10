use ndarray::{Array4, ArrayView4};
use rand::Rng;
use rand_distr::StandardNormal;

use super::{Euler, Scheduler, Timesteps};

#[derive(Debug, Default)]
pub struct EulerAncestral(Euler);

impl Scheduler for EulerAncestral {
    #[inline]
    fn step(
        &mut self,
        step: usize,
        steps: &Timesteps,
        sample: ArrayView4<'_, f32>,
        output: ArrayView4<'_, f32>,
        rng: &mut impl Rng,
    ) -> Array4<f32> {
        let from = steps.sigmas[step];
        let to = steps.sigmas[step + 1];
        let up = to * (1. - (to.powi(2) / from.powi(2))).powf(0.5);
        let down = to.powi(2) / from;
        let noise =
            Array4::<f32>::from_shape_simple_fn(output.dim(), || rng.sample(StandardNormal));
        &sample + &output * (down - from) + noise * up
    }

    #[inline]
    fn add_noise(
        &self,
        step: usize,
        steps: &Timesteps,
        sample: ArrayView4<'_, f32>,
        noise: ArrayView4<'_, f32>,
    ) -> Array4<f32> {
        self.0.add_noise(step, steps, sample, noise)
    }

    #[inline]
    fn timesteps(&mut self, num_inference_steps: usize) -> Timesteps {
        self.0.timesteps(num_inference_steps)
    }

    #[inline]
    fn init_scale_multiplier(&self) -> f32 {
        self.0.init_scale_multiplier()
    }

    #[inline]
    fn scale_multiplier(&self, sigma: f32) -> f32 {
        self.0.scale_multiplier(sigma)
    }
}
