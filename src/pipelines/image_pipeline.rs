use std::time::Instant;

use image::DynamicImage;
use ndarray::concatenate;
use ndarray::prelude::*;
use ort::tensor::{FromArray, InputTensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use super::{PipelineError, PipelineInput, PipelineMode, Prompt};
use crate::checkpoint::Checkpoint;
use crate::schedulers::Scheduler;
use crate::tokenizers::base::BaseEncoder;
use crate::tokenizers::lpw::LongPromptEncoder;
use crate::tokenizers::EncodedPrompts;

#[derive(Debug)]
pub struct ImagePipeline {
    pub steps: usize,
    pub seed: u64,
    pub prompts: Vec<Prompt>,
    pub guidance_scale: f32,
    pub long_prompt_weighting: bool,
    pub input: PipelineInput,
}

impl ImagePipeline {
    const MAX_EMBEDDING_MULTIPLES: usize = 3;

    pub fn with_prompt(mut self, prompt: Prompt) -> Self {
        self.prompts.push(prompt);
        self
    }

    pub fn with_input_image(mut self, image: DynamicImage, denoise: f32) -> Self {
        self.input = PipelineInput::Image { image, denoise };
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    pub fn with_no_long_prompt_weighting(mut self) -> Self {
        self.long_prompt_weighting = false;
        self
    }

    pub fn with_guidance_scale(mut self, guidance_scale: f32) -> Self {
        self.guidance_scale = guidance_scale;
        self
    }

    fn is_classifier_free_guidance(&self) -> bool {
        self.guidance_scale != 1.0
    }

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
        let positive = self.prompts.iter().map(|p| &p.positive);
        let negative = self.prompts.iter().map(|p| &p.negative);

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

        let timesteps = scheduler.timesteps(self.steps);

        let (mut latent, width, height) = match &self.input {
            &PipelineInput::EmptyLatent { width, height } => {
                let latent = Array4::<f32>::from_shape_simple_fn(
                    (self.prompts.len(), 4, height / 8, width / 8),
                    || rng.sample(StandardNormal),
                ) * scheduler.init_scale_multiplier();
                (latent, width, height)
            }
            PipelineInput::Image { image, denoise } => {
                let encoded = checkpoint.encode_latents(image, self.prompts.len())?;
                let noise = Array4::<f32>::from_shape_simple_fn(encoded.latents.dim(), || {
                    rng.sample(StandardNormal)
                });

                let start_step = self.steps - (self.steps as f32 * denoise) as usize;
                let latent = scheduler.add_noise(
                    start_step,
                    &timesteps,
                    encoded.latents.view(),
                    noise.view(),
                );
                (latent, encoded.width, encoded.height)
            }
        };

        let now = Instant::now();

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
            let output = Mode::into_f32_array(output.view().clone().into_dimensionality()?.into());
            let output = if self.is_classifier_free_guidance() {
                let uncond = output.index_axis(Axis(0), 0);
                let text = output.index_axis(Axis(0), 1);
                (&uncond + self.guidance_scale * (&text - &uncond)).insert_axis(Axis(0))
            } else {
                output.to_owned()
            };

            latent = scheduler.step(i, &timesteps, latent.view(), output.view(), &mut rng);
        }

        checkpoint.decode_latents(width as u32, height as u32, latent.view())
    }
}

impl Default for ImagePipeline {
    fn default() -> Self {
        Self {
            guidance_scale: 7.5,
            steps: 20,
            seed: 0,
            prompts: vec![],
            long_prompt_weighting: true,
            input: PipelineInput::EmptyLatent {
                width: 512,
                height: 512,
            },
        }
    }
}
