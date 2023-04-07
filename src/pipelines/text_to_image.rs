use image::DynamicImage;
use ndarray::{concatenate, Array4, Axis};
use ort::tensor::{FromArray, InputTensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::{PipelineError, PipelineMode, Prompt};
use crate::checkpoint::{Checkpoint, TensorHolder};
use crate::schedulers::Scheduler;

#[derive(Debug)]
pub struct TextToImage {
    pub height: u32,
    pub width: u32,
    pub guidance_scale: f32,
    pub steps: usize,
    pub seed: u64,
    pub batch: Vec<Prompt>,
}

impl TextToImage {
    fn is_classifier_free_guidance(&self) -> bool {
        self.guidance_scale != 1.0
    }
}

impl Default for TextToImage {
    fn default() -> Self {
        Self {
            height: 512,
            width: 512,
            guidance_scale: 7.5,
            steps: 20,
            seed: 0,
            batch: vec![],
        }
    }
}

impl TextToImage {
    pub fn execute<Mode>(
        &self,
        pipeline: &Checkpoint<Mode>,
        scheduler: &mut impl Scheduler,
    ) -> Result<Vec<DynamicImage>, PipelineError>
    where
        Mode: PipelineMode,
        InputTensor: FromArray<Mode::Float>,
    {
        let mut rng = StdRng::seed_from_u64(self.seed);

        let negative = pipeline.encode_prompt(self.batch.iter().map(|p| &p.negative))?;
        let positive = pipeline.encode_prompt(self.batch.iter().map(|p| &p.positive))?;
        let prompt = if self.is_classifier_free_guidance() {
            concatenate![Axis(0), *negative.view(), *positive.view()]
        } else {
            positive.view().to_owned()
        };

        let mut latent = Array4::<f32>::from_shape_simple_fn(
            (
                self.batch.len(),
                4,
                self.height as usize / 8,
                self.width as usize / 8,
            ),
            || rng.sample(StandardNormal),
        ) * scheduler.sigma_multiplier();

        let timesteps = scheduler.timesteps(self.steps);
        for (i, (&ts, &sigma)) in timesteps
            .timesteps
            .iter()
            .zip(&timesteps.sigmas)
            .enumerate()
        {
            println!("Step {}/{}", i, timesteps.timesteps.len());

            let latent_input = if self.is_classifier_free_guidance() {
                concatenate![Axis(0), latent, latent] * scheduler.scale_multiplier(sigma)
            } else {
                &latent * scheduler.scale_multiplier(sigma)
            };
            let output = pipeline.eval_unet(latent_input, ts, prompt.view().to_owned())?;
            let output = output
                .view()
                .map(|&f: &Mode::Float| f.into())
                .into_dimensionality()?;
            let output = if self.is_classifier_free_guidance() {
                let uncond = output.index_axis(Axis(0), 0);
                let text = output.index_axis(Axis(0), 1);
                (&uncond + self.guidance_scale * (&text - &uncond)).insert_axis(Axis(0))
            } else {
                output.to_owned()
            };

            latent = scheduler.step(i, &timesteps, latent.view(), output.view(), &mut rng);
        }

        pipeline.decode_latents(self.width, self.height, latent.view())
    }
}
