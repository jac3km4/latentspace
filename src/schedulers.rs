use ndarray::{Array1, Array4, ArrayView4};
use rand::Rng;

mod euler;
mod euler_ancestral;
mod lms;

pub use euler::Euler;
pub use euler_ancestral::EulerAncestral;
pub use lms::Lms;

pub trait Scheduler {
    fn step(
        &mut self,
        step: usize,
        steps: &Timesteps,
        sample: ArrayView4<'_, f32>,
        output: ArrayView4<'_, f32>,
        rng: &mut impl Rng,
    ) -> Array4<f32>;

    fn add_noise(
        &self,
        step: usize,
        steps: &Timesteps,
        sample: ArrayView4<'_, f32>,
        noise: ArrayView4<'_, f32>,
    ) -> Array4<f32>;

    fn timesteps(&mut self, num_inference_steps: usize) -> Timesteps;
    fn init_scale_multiplier(&self) -> f32;
    fn scale_multiplier(&self, sigma: f32) -> f32;
}

#[derive(Debug)]
pub struct Timesteps {
    pub sigmas: Vec<f32>,
    pub timesteps: Array1<f32>,
}
