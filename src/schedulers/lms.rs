use std::{iter, mem};

use ndarray::{s, Array1, Array4, ArrayView4};
use rand::Rng;

use super::{Euler, Scheduler, Timesteps};

#[derive(Debug)]
pub struct Lms {
    inner: Euler,
    order: usize,
    outputs: Vec<Array4<f32>>,
}

impl Scheduler for Lms {
    #[inline]
    fn step(
        &mut self,
        step: usize,
        steps: &Timesteps,
        sample: ArrayView4<'_, f32>,
        output: ArrayView4<'_, f32>,
        _rng: &mut impl Rng,
    ) -> Array4<f32> {
        let mut result = sample.to_owned();
        let outputs = mem::take(&mut self.outputs);
        for (i, arr) in iter::once(output.to_owned())
            .chain(outputs.into_iter().take(self.order - 1))
            .enumerate()
        {
            let x = Array1::linspace(steps.sigmas[step], steps.sigmas[step + 1], 81);
            let mut y: Array1<f32> = Array1::ones(81);

            for j in 0..self.order.min(i) {
                y = y * (&x - steps.sigmas[step - j])
                    / (steps.sigmas[step - i] - steps.sigmas[step - j]);
            }

            let delta_x = (steps.sigmas[step + 1] - steps.sigmas[step]) / 80.0;
            let coeff = (y.slice(s![0..80]).sum() + y[80] / 2.0) * delta_x;

            result = result + coeff * &arr;
            self.outputs.push(arr);
        }
        result
    }

    #[inline]
    fn add_noise(
        &self,
        step: usize,
        steps: &Timesteps,
        sample: ArrayView4<'_, f32>,
        noise: ArrayView4<'_, f32>,
    ) -> Array4<f32> {
        self.inner.add_noise(step, steps, sample, noise)
    }

    #[inline]
    fn timesteps(&mut self, num_inference_steps: usize) -> Timesteps {
        self.inner.timesteps(num_inference_steps)
    }

    #[inline]
    fn init_scale_multiplier(&self) -> f32 {
        self.inner.init_scale_multiplier()
    }

    #[inline]
    fn scale_multiplier(&self, sigma: f32) -> f32 {
        self.inner.scale_multiplier(sigma)
    }
}

impl Default for Lms {
    fn default() -> Self {
        Self {
            inner: Euler::default(),
            order: 4,
            outputs: vec![],
        }
    }
}
