use std::fmt;

use half::f16;
use ndarray::{CowArray, Dimension};
use ort::tensor::{FromArray, InputTensor, TensorDataToType};
use ort::OrtError;
use thiserror::Error;

mod text_to_image;
pub use text_to_image::TextToImage;

pub trait PipelineMode: Default {
    type Float: Copy + Into<f32> + TensorDataToType + 'static;

    fn create_tensor<D: Dimension>(array: CowArray<'_, f32, D>) -> InputTensor;
}

#[derive(Debug, Default)]
pub struct F32Mode;

impl PipelineMode for F32Mode {
    type Float = f32;

    #[inline]
    fn create_tensor<D: Dimension>(array: CowArray<'_, f32, D>) -> InputTensor {
        InputTensor::from_array(array.into_owned().into_dyn())
    }
}

#[derive(Debug, Default)]
pub struct F16Mode;

impl PipelineMode for F16Mode {
    type Float = f16;

    #[inline]
    fn create_tensor<D: Dimension>(array: CowArray<'_, f32, D>) -> InputTensor {
        InputTensor::from_array(array.map(|&f| f16::from_f32(f)).into_dyn())
    }
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
