use std::iter;

use ndarray::prelude::*;
use ort::tensor::{FromArray, InputTensor};

use super::clip::Token;
use super::EncodedPrompts;
use crate::pipelines::{PipelineError, PipelineMode, Stage};
use crate::Checkpoint;

#[derive(Debug)]
pub(crate) struct LongPromptEncoder<'c, Mode> {
    checkpoint: &'c Checkpoint<Mode>,
    max_embeddings_multiples: usize,
}

impl<'c, Mode: PipelineMode> LongPromptEncoder<'c, Mode> {
    pub fn new(checkpoint: &'c Checkpoint<Mode>, max_embeddings_multiples: usize) -> Self {
        Self {
            checkpoint,
            max_embeddings_multiples,
        }
    }

    pub fn encode(
        &self,
        prompts: impl IntoIterator<Item = impl AsRef<str>>,
        neg_prompts: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<EncodedPrompts<Mode::Float>, PipelineError> {
        let max_len =
            (self.checkpoint.tokenizer().max_len() - 2) * self.max_embeddings_multiples + 2;
        let (mut tokens, mut weights) = self.tokenize_with_weights(prompts, max_len - 2);
        let (mut neg_tokens, mut neg_weights) =
            self.tokenize_with_weights(neg_prompts, max_len - 2);

        let max_len = tokens
            .iter()
            .chain(&neg_tokens)
            .map(Vec::len)
            .max()
            .unwrap_or(0);
        let max_chunks = self
            .max_embeddings_multiples
            .min((max_len - 1) / (self.checkpoint.tokenizer().max_len() - 2) + 1)
            .max(1);
        let max_len = (self.checkpoint.tokenizer().max_len() - 2) * max_chunks + 2;

        self.pad_tokens(&mut tokens, &mut weights, max_len);
        self.pad_tokens(&mut neg_tokens, &mut neg_weights, max_len);

        let shape = (tokens.len(), max_len);
        let embedding = self.encode_tokens(Array2::from_shape_vec(shape, tokens.concat())?)?;
        let neg_embedding =
            self.encode_tokens(Array2::from_shape_vec(shape, neg_tokens.concat())?)?;

        let positive = apply_weights(
            Mode::into_f32_array(embedding.into()).view(),
            Array2::from_shape_vec(shape, weights.concat())?.view(),
        );
        let negative = apply_weights(
            Mode::into_f32_array(neg_embedding.into()).view(),
            Array2::from_shape_vec(shape, neg_weights.concat())?.view(),
        );
        Ok(EncodedPrompts {
            positive: Mode::from_f32_array(positive.into()).into_owned(),
            negative: Mode::from_f32_array(negative.into()).into_owned(),
        })
    }

    fn tokenize_with_weights(
        &self,
        prompts: impl IntoIterator<Item = impl AsRef<str>>,
        max_length: usize,
    ) -> (Vec<Vec<Token>>, Vec<Vec<f32>>) {
        prompts
            .into_iter()
            .map(|prompt| {
                let mut tokens = vec![self.checkpoint.tokenizer().start_token()];
                let mut weights = vec![1.];

                for weighted in ChunkParser::new(prompt.as_ref()).consume() {
                    let len = tokens.len();
                    self.checkpoint
                        .tokenizer()
                        .tokenize_into(weighted.text, &mut tokens);
                    weights.extend(iter::repeat(weighted.weight).take(tokens.len() - len));

                    if tokens.len() > max_length {
                        tokens.truncate(max_length);
                        weights.truncate(max_length);
                        break;
                    }
                }
                (tokens, weights)
            })
            .unzip()
    }

    fn pad_tokens(&self, tokens: &mut [Vec<Token>], weights: &mut [Vec<f32>], max_length: usize) {
        let end_token = self.checkpoint.tokenizer().end_token();
        for (tokens, weights) in tokens.iter_mut().zip(weights) {
            tokens.extend(iter::repeat(end_token).take(max_length - tokens.len()));
            weights.extend(iter::repeat(1.).take(max_length - weights.len()));
        }
    }

    fn encode_tokens(&self, tokens: Array2<Token>) -> Result<Array3<Mode::Float>, PipelineError> {
        let chunk_size = self.checkpoint.tokenizer().max_len();
        let chunk_count = (tokens.shape()[1] - 2) / (chunk_size - 2);
        let output_size = self
            .checkpoint
            .text_encoder()
            .outputs
            .first()
            .and_then(|out| Some((*out.dimensions.get(2)?)? as usize))
            .unwrap_or(768);
        let mut outputs = Array3::default((1, 0, output_size));

        for i in 0..chunk_count {
            let start = i * (chunk_size - 2);
            let mut chunk = tokens.slice(s![.., start..start + chunk_size]).to_owned();
            chunk
                .index_axis_mut(Axis(1), 0)
                .assign(&tokens.index_axis(Axis(1), 0));
            chunk
                .index_axis_mut(Axis(1), chunk.dim().1 - 1)
                .assign(&tokens.index_axis(Axis(1), tokens.dim().1 - 1));
            let results = self
                .checkpoint
                .text_encoder()
                .run([InputTensor::from_array(chunk.into_dyn())])?;
            let [chunk_embeddings, _] = &results[..] else {
                return Err(PipelineError::OutputError(Stage::Text));
            };
            let arr = chunk_embeddings.try_extract()?;
            let arr = arr.view().clone().into_dimensionality()?;
            let arr = match i {
                _ if chunk_count == 1 => arr,
                0 => arr.slice(s![.., ..-1, ..]),
                i if i == chunk_count - 1 => arr.slice(s![.., 1.., ..]),
                #[allow(clippy::reversed_empty_ranges)]
                _ => arr.slice(s![.., 1..-1, ..]),
            };
            outputs.append(Axis(1), arr)?;
        }
        Ok(outputs)
    }
}

fn apply_weights(text: ArrayView3<'_, f32>, weights: ArrayView2<'_, f32>) -> Array3<f32> {
    let mean0 = text.mean_axis(Axis(2)).unwrap().mean_axis(Axis(1)).unwrap();
    let text = &text * &weights.insert_axis(Axis(2));
    let mean1 = text.mean_axis(Axis(2)).unwrap().mean_axis(Axis(1)).unwrap();
    text * (mean0 / mean1)
}

#[derive(Debug, PartialEq)]
struct Weighted<'a> {
    text: &'a str,
    weight: f32,
}

#[derive(Debug)]
struct ChunkParser<'a> {
    tokens: iter::Peekable<TokenIter<'a>>,
    output: Vec<Weighted<'a>>,
    stack: Vec<f32>,
}

impl<'a> ChunkParser<'a> {
    const ROUND_BRACKET_MULTIPLIER: f32 = 1.1;
    const SQUARE_BRACKET_MULTIPLIER: f32 = 1. / 1.1;

    fn new(text: &'a str) -> Self {
        Self {
            tokens: TokenIter::new(text).peekable(),
            output: vec![],
            stack: vec![],
        }
    }

    fn consume(mut self) -> Vec<Weighted<'a>> {
        for token in self.tokens {
            match (token, self.stack.split_last()) {
                ("(", _) => {
                    self.stack.push(Self::ROUND_BRACKET_MULTIPLIER);
                }
                ("[", _) => {
                    self.stack.push(Self::SQUARE_BRACKET_MULTIPLIER);
                }
                ("]" | ")", _) => {
                    self.stack.pop();
                }
                (body, Some((&default, stack))) => {
                    let (text, weight) = body
                        .split_once(':')
                        .and_then(|(text, num)| Some((text, num.parse().ok()?)))
                        .unwrap_or((body, default));
                    let weight = weight * stack.iter().product::<f32>();
                    self.output.push(Weighted { text, weight });
                }
                (text, _) => self.output.push(Weighted { text, weight: 1. }),
            }
        }
        self.output
    }
}

#[derive(Debug)]
struct TokenIter<'a> {
    text: &'a str,
}

impl<'a> TokenIter<'a> {
    fn new(text: &'a str) -> Self {
        Self { text }
    }

    #[inline]
    fn split_off(&mut self, n: usize) -> &'a str {
        let (ret, rest) = self.text.split_at(n);
        self.text = rest;
        ret
    }
}

impl<'a> Iterator for TokenIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self.text.as_bytes() {
            [] => None,
            [b'\\', b'\\' | b'(' | b')' | b'[' | b']', ..] => Some(self.split_off(2)),
            [b'(' | b'[' | b')' | b']', ..] => Some(self.split_off(1)),
            _ => {
                let n = self
                    .text
                    .find(|c| c == '(' || c == '[' || c == ')' || c == ']' || c == '\\')
                    .unwrap_or(self.text.len());
                Some(self.split_off(n))
            }
        }
    }
}

#[cfg(test)]
mod test {

    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn adasd() {
        assert_eq!(
            ChunkParser::new("a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).").consume(),
            &[
                Weighted {
                    text: "a ",
                    weight: 1.0
                },
                Weighted {
                    text: "house",
                    weight: 1.573
                },
                Weighted {
                    text: " ",
                    weight: 1.1
                },
                Weighted {
                    text: "on",
                    weight: 1.0
                },
                Weighted {
                    text: " a ",
                    weight: 1.1
                },
                Weighted {
                    text: "hill",
                    weight: 0.55,
                },
                Weighted {
                    text: ", sun, ",
                    weight: 1.1
                },
                Weighted {
                    text: "sky",
                    weight: 1.4641001
                },
                Weighted {
                    text: ".",
                    weight: 1.1
                }
            ]
        );
    }
}
