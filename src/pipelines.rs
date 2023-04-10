use std::fmt;

use half::f16;
use image::DynamicImage;
use ndarray::{CowArray, Dimension};
use ort::tensor::{InputTensor, TensorDataToType};
use ort::OrtError;
use thiserror::Error;

mod image_pipeline;
pub use image_pipeline::ImagePipeline;

pub trait PipelineMode: Default {
    type Float: Copy + Default + Into<f32> + TensorDataToType + 'static;

    fn from_f32_array<D: Dimension>(array: CowArray<'_, f32, D>) -> CowArray<'_, Self::Float, D>;
    fn into_f32_array<D: Dimension>(array: CowArray<'_, Self::Float, D>) -> CowArray<'_, f32, D>;
    fn create_tensor<D: Dimension>(cow: CowArray<'_, f32, D>) -> InputTensor;
}

#[derive(Debug, Default)]
pub struct Fp32Mode;

impl PipelineMode for Fp32Mode {
    type Float = f32;

    #[inline]
    fn from_f32_array<D: Dimension>(array: CowArray<'_, f32, D>) -> CowArray<'_, Self::Float, D> {
        array
    }

    #[inline]
    fn into_f32_array<D: Dimension>(array: CowArray<'_, Self::Float, D>) -> CowArray<'_, f32, D> {
        array
    }

    #[inline]
    fn create_tensor<D: Dimension>(cow: CowArray<'_, f32, D>) -> InputTensor {
        InputTensor::FloatTensor(cow.into_owned().into_dyn())
    }
}

#[derive(Debug, Default)]
pub struct Fp16Mode;

impl PipelineMode for Fp16Mode {
    type Float = f16;

    fn from_f32_array<D: Dimension>(array: CowArray<'_, f32, D>) -> CowArray<'_, Self::Float, D> {
        array.map(|&x| f16::from_f32(x)).into()
    }

    fn into_f32_array<D: Dimension>(array: CowArray<'_, Self::Float, D>) -> CowArray<'_, f32, D> {
        array.map(|&x| x.into()).into()
    }

    fn create_tensor<D: Dimension>(cow: CowArray<'_, f32, D>) -> InputTensor {
        InputTensor::Float16Tensor(Self::from_f32_array(cow).into_owned().into_dyn())
    }
}

#[derive(Debug)]
pub enum PipelineInput {
    EmptyLatent { width: usize, height: usize },
    Image { image: DynamicImage, denoise: f32 },
}

#[derive(Debug, Clone, Copy)]
pub enum Stage {
    Text,
    VaeEncode,
    VaeDecode,
    Unet,
    Safety,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "text encoder"),
            Self::VaeEncode => write!(f, "vae encoder"),
            Self::VaeDecode => write!(f, "vae decoder"),
            Self::Unet => write!(f, "unet"),
            Self::Safety => write!(f, "safety checker"),
        }
    }
}

#[derive(Debug)]
pub struct Prompt {
    pub positive: String,
    pub negative: String,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PipelineError {
    #[error("ORT error: {0}")]
    Ort(#[from] OrtError),
    #[error("unexpected tensor shape: {0}")]
    ShapeError(#[from] ndarray::ShapeError),
    #[error("unexpected output error in {0}")]
    OutputError(Stage),
}
