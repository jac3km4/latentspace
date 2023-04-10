use ndarray::{Array2, Array3};
use ort::tensor::{FromArray, InputTensor};

use super::EncodedPrompts;
use crate::pipelines::{PipelineError, PipelineMode, Stage};
use crate::Checkpoint;

#[derive(Debug)]
pub(crate) struct BaseEncoder<'c, Mode> {
    checkpoint: &'c Checkpoint<Mode>,
}

impl<'c, Mode: PipelineMode> BaseEncoder<'c, Mode> {
    pub fn new(checkpoint: &'c Checkpoint<Mode>) -> Self {
        Self { checkpoint }
    }

    pub fn encode(
        &self,
        prompts: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl AsRef<str>>>,
        neg_prompts: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl AsRef<str>>>,
    ) -> Result<EncodedPrompts<Mode::Float>, PipelineError> {
        let positive = self.encode_batch(prompts)?;
        let negative = self.encode_batch(neg_prompts)?;
        Ok(EncodedPrompts { positive, negative })
    }

    fn encode_batch(
        &self,
        batch: impl IntoIterator<IntoIter = impl ExactSizeIterator<Item = impl AsRef<str>>>,
    ) -> Result<Array3<Mode::Float>, PipelineError> {
        let it = batch.into_iter();
        let len = it.len();
        let tokens: Vec<_> = it
            .flat_map(|str| self.checkpoint.tokenizer().tokenize(str.as_ref()))
            .collect();
        let array = Array2::from_shape_vec((len, self.checkpoint.tokenizer().max_len()), tokens)?;
        let result = self
            .checkpoint
            .text_encoder()
            .run([InputTensor::from_array(array.into_dyn())])?;
        let [tensor, _] = &result[..] else {
            return Err(PipelineError::OutputError(Stage::Text))
        };
        Ok(tensor
            .try_extract()?
            .view()
            .to_owned()
            .into_dimensionality()?)
    }
}
