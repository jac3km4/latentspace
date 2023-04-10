use std::time::Instant;

use image::DynamicImage;
use ndarray::concatenate;
use ndarray::prelude::*;
use ort::tensor::{FromArray, InputTensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::{PipelineError, PipelineMode, Prompt};
use crate::checkpoint::Checkpoint;
use crate::schedulers::Scheduler;
use crate::tokenizers::base::BaseEncoder;
use crate::tokenizers::lpw::LongPromptEncoder;
use crate::tokenizers::EncodedPrompts;

#[derive(Debug)]
pub struct TextToImage {
    pub height: u32,
    pub width: u32,
    pub guidance_scale: f32,
    pub steps: usize,
    pub seed: u64,
    pub batch: Vec<Prompt>,
    pub long_prompt_weighting: bool,
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
            long_prompt_weighting: true,
        }
    }
}

impl TextToImage {
    const MAX_EMBEDDING_MULTIPLES: usize = 3;

    pub fn execute<Mode>(
        &self,
        checkpoint: &Checkpoint<Mode>,
        scheduler: &mut impl Scheduler,
    ) -> Result<Vec<DynamicImage>, PipelineError>
    where
        Mode: PipelineMode,
        InputTensor: FromArray<Mode::Float>,
    {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let positive = self.batch.iter().map(|p| &p.positive);
        let negative = self.batch.iter().map(|p| &p.negative);

        let EncodedPrompts { positive, negative } = if self.long_prompt_weighting {
            LongPromptEncoder::new(checkpoint, Self::MAX_EMBEDDING_MULTIPLES)
                .encode(positive, negative)?
        } else {
            BaseEncoder::new(checkpoint).encode(positive, negative)?
        };

        let prompt = if self.is_classifier_free_guidance() {
            concatenate![Axis(0), negative.view(), positive.view()]
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
        ) * scheduler.init_scale_multiplier();

        let now = Instant::now();

        let timesteps = scheduler.timesteps(self.steps);
        for (i, (&ts, &sigma)) in timesteps
            .timesteps
            .iter()
            .zip(&timesteps.sigmas)
            .enumerate()
        {
            println!(
                "Step {}/{} ({:.2}s)",
                i,
                timesteps.timesteps.len(),
                now.elapsed().as_secs_f32()
            );

            let latent_input = if self.is_classifier_free_guidance() {
                concatenate![Axis(0), latent, latent] * scheduler.scale_multiplier(sigma)
            } else {
                &latent * scheduler.scale_multiplier(sigma)
            };
            let output = checkpoint.eval_unet(latent_input, ts, prompt.to_owned().into_dyn())?;
            let output =
                Mode::into_f32_array(output.view().clone().into()).into_dimensionality()?;
            let output = if self.is_classifier_free_guidance() {
                let uncond = output.index_axis(Axis(0), 0);
                let text = output.index_axis(Axis(0), 1);
                (&uncond + self.guidance_scale * (&text - &uncond)).insert_axis(Axis(0))
            } else {
                output.to_owned()
            };

            latent = scheduler.step(i, &timesteps, latent.view(), output.view(), &mut rng);
        }

        checkpoint.decode_latents(self.width, self.height, latent.view())
    }
}
