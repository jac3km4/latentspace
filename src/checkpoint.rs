use std::path::Path;
use std::sync::Arc;

use image::{DynamicImage, Rgb32FImage};
use ndarray::prelude::*;
use ort::tensor::{FromArray, InputTensor, OrtOwnedTensor};
use ort::OrtError;
use thiserror::Error;

use crate::config::DeviceConfig;
use crate::pipelines::{PipelineError, PipelineMode, Stage};
use crate::tokenizers::clip::{ClipTokenizer, TokenizerError};

#[derive(Debug)]
pub struct Checkpoint<Mode> {
    clip: ClipTokenizer,
    _vae_encoder: ort::Session,
    vae_decoder: ort::Session,
    text_encoder: ort::Session,
    unet: ort::Session,
    _safety_checker: Option<ort::Session>,
    _mode: Mode,
}

impl<Mode: PipelineMode> Checkpoint<Mode> {
    pub fn load(
        env: Arc<ort::Environment>,
        devices: &DeviceConfig,
        model_dir: impl AsRef<Path>,
    ) -> Result<Self, CheckpointLoadError> {
        let clip = ClipTokenizer::open(model_dir.as_ref().join("tokenizer"))?;
        let text_encoder = ort::SessionBuilder::new(&env)?
            .with_execution_providers(devices.text_encoder.into_execution_providers())?
            .with_model_from_file(model_dir.as_ref().join("text_encoder").join("model.onnx"))?;
        let _vae_encoder = ort::SessionBuilder::new(&env)?
            .with_execution_providers(devices.vae_encoder.into_execution_providers())?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_dir.as_ref().join("vae_encoder").join("model.onnx"))?;
        let vae_decoder = ort::SessionBuilder::new(&env)?
            .with_execution_providers(devices.vae_decoder.into_execution_providers())?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_dir.as_ref().join("vae_decoder").join("model.onnx"))?;
        let unet = ort::SessionBuilder::new(&env)?
            .with_execution_providers(devices.unet.into_execution_providers())?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_model_from_file(model_dir.as_ref().join("unet").join("model.onnx"))?;

        Ok(Self {
            clip,
            vae_decoder,
            text_encoder,
            unet,
            _vae_encoder,
            _safety_checker: None, // TODO: add safety checker
            _mode: Mode::default(),
        })
    }

    pub(crate) fn eval_unet(
        &self,
        latent: Array4<f32>,
        timestep: f32,
        prompt: ArrayD<Mode::Float>,
    ) -> Result<OrtOwnedTensor<'_, Mode::Float, IxDyn>, PipelineError>
    where
        InputTensor: FromArray<Mode::Float>,
    {
        let noise = self.unet.run([
            Mode::create_tensor(latent.into()),
            Mode::create_tensor(Array1::from_elem(1, timestep).into()),
            InputTensor::from_array(prompt.into_dyn()),
        ])?;
        let [output] = &noise[..] else {
            return Err(PipelineError::OutputError(Stage::Unet));
        };
        let output = output.try_extract()?;
        Ok(output)
    }

    pub(crate) fn decode_latents(
        &self,
        width: u32,
        height: u32,
        latent: ArrayView4<'_, f32>,
    ) -> Result<Vec<DynamicImage>, PipelineError> {
        let latent = &latent / 0.18215;
        latent
            .axis_iter(Axis(0))
            .map(|batch| {
                let chunk = batch.insert_axis(Axis(0));
                let result = self.vae_decoder.run([Mode::create_tensor(chunk.into())])?;
                let [tensor] = &result[..] else {
                    return Err(PipelineError::OutputError(Stage::VaeDecode))
                };
                let pixels = tensor
                    .try_extract()?
                    .view()
                    .clone()
                    .into_dimensionality()?
                    .map(|&f: &Mode::Float| (f.into() / 2. + 0.5))
                    .permuted_axes([0, 2, 3, 1])
                    .into_iter()
                    .collect();
                Ok(Rgb32FImage::from_raw(width, height, pixels)
                    .expect("image creation failed")
                    .into())
            })
            .collect()
    }
}

impl<M> Checkpoint<M> {
    pub(crate) fn text_encoder(&self) -> &ort::Session {
        &self.text_encoder
    }

    pub(crate) fn tokenizer(&self) -> &ClipTokenizer {
        &self.clip
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointLoadError {
    #[error("ORT error: {0}")]
    Ort(#[from] OrtError),
    #[error("{0}")]
    Tokenizer(#[from] TokenizerError),
}
