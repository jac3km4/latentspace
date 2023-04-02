use ndarray::{Array1, Array4, ArrayView4};
use rand::Rng;

mod euler;
pub use euler::Euler;

pub trait Scheduler {
    fn step(
        &mut self,
        index: usize,
        steps: &Timesteps,
        sample: ArrayView4<'_, f32>,
        output: ArrayView4<'_, f32>,
        rng: &mut impl Rng,
    ) -> Array4<f32>;

    fn timesteps(&mut self, num_inference_steps: usize) -> Timesteps;
    fn sigma_multiplier(&self) -> f32;
    fn scale_multiplier(&self, sigma: f32) -> f32;
}

#[derive(Debug)]
pub struct Timesteps {
    pub sigmas: Vec<f32>,
    pub timesteps: Array1<f32>,
}
