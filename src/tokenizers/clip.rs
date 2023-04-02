use std::collections::HashMap;
use std::fs::File;
use std::io::BufRead;
use std::path::Path;
use std::{io, iter, mem};

use ecow::EcoString;
use serde::Deserialize;
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

type Token = i32;

#[derive(Debug)]
pub struct ClipTokenizer {
    start_token: Token,
    end_token: Token,
    pad_token: Token,
    max_length: usize,
    vocab: HashMap<EcoString, Token>,
    merges: HashMap<(EcoString, EcoString), Token>,
}

impl ClipTokenizer {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let config_file = File::open(directory.as_ref().join("tokenizer_config.json"))?;
        let config: TokenizerConfig = serde_json::from_reader(config_file)?;

        if config.tokenizer_class != "CLIPTokenizer" {
            return Err(TokenizerError::UnsupportedTokenizer(config.tokenizer_class));
        }

        let source = io::BufReader::new(File::open(directory.as_ref().join("vocab.json"))?);
        let vocab: HashMap<EcoString, Token> = serde_json::from_reader(source)?;

        let merges = io::BufReader::new(File::open(directory.as_ref().join("merges.txt"))?)
            .lines()
            .enumerate()
            .map(|(i, line)| {
                let line = line?;
                let (lhs, rhs) = line.split_once(' ').ok_or(TokenizerError::InvalidMerges)?;
                let i = i.try_into().map_err(|_| TokenizerError::InvalidMerges)?;
                Ok(((lhs.into(), rhs.into()), i))
            })
            .collect::<Result<_, TokenizerError>>()?;

        let start_token = *vocab
            .get(config.bos_token.content.as_str())
            .ok_or(TokenizerError::MissingStartToken)?;
        let end_token = *vocab
            .get(config.eos_token.content.as_str())
            .ok_or(TokenizerError::MissingEndToken)?;
        let pad_token = *vocab
            .get(config.pad_token.as_str())
            .ok_or(TokenizerError::MissingPadToken)?;

        Ok(Self {
            start_token,
            end_token,
            pad_token,
            vocab,
            merges,
            max_length: config.model_max_length,
        })
    }

    pub fn encode(&self, text: impl AsRef<str>) -> Vec<Token> {
        let str: String = text.as_ref().nfc().flat_map(char::to_lowercase).collect();
        let mut tokens = vec![self.start_token];
        for chunk in ChunkIter::new(&str) {
            let chunk = chunk
                .as_bytes()
                .iter()
                .map(|&c| TABLE[usize::from(c)])
                .collect::<EcoString>();
            let words = self
                .bpe(&chunk)
                .into_iter()
                .filter_map(|word| self.vocab.get(&word));
            tokens.extend(words);
        }
        tokens.push(self.end_token);
        tokens.truncate(self.max_length);

        let pad_length = self.max_length - tokens.len();
        tokens.extend(iter::repeat(self.pad_token).take(pad_length));
        tokens
    }

    fn bpe(&self, chunk: &str) -> Vec<EcoString> {
        let mut words: Vec<_> = chunk.chars().map(EcoString::from).collect();
        words.last_mut().iter_mut().for_each(|s| s.push_str("</w>"));
        let mut scratch: Vec<EcoString> = Vec::with_capacity(words.len());

        while words.len() > 1 {
            let Some(((fst, snd), _)) = words
                .iter()
                .zip(words.iter().skip(1))
                .filter_map(|(fst, snd)| self.merges.get_key_value(&(fst.clone(), snd.clone())))
                .min_by_key(|(_, &v)| v) else { break };

            for word in words.drain(..) {
                match scratch.last_mut() {
                    Some(last) if &word == snd && last == fst => last.push_str(snd),
                    _ => scratch.push(word),
                }
            }
            mem::swap(&mut words, &mut scratch);
        }

        words
    }

    pub fn max_length(&self) -> usize {
        self.max_length
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TokenizerError {
    #[error("missing start token in the vocabulary")]
    MissingStartToken,
    #[error("missing end token in the vocabulary")]
    MissingEndToken,
    #[error("missing pad token in the vocabulary")]
    MissingPadToken,
    #[error("failed to parse tokenizer JSON config: {0}")]
    InvalidConfig(#[from] serde_json::Error),
    #[error("unsupported tokenizer class: {0}")]
    UnsupportedTokenizer(String),
    #[error("invalid merges file")]
    InvalidMerges,
    #[error("I/O error when loading the tokenizer: {0}")]
    Io(#[from] io::Error),
}

#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    tokenizer_class: String,
    bos_token: TokenConfig,
    eos_token: TokenConfig,
    pad_token: String,
    model_max_length: usize,
}

#[derive(Debug, Deserialize)]
struct TokenConfig {
    content: String,
}

struct ChunkIter<'a> {
    inner: &'a str,
}

impl<'a> ChunkIter<'a> {
    fn new(inner: &'a str) -> Self {
        Self { inner }
    }

    #[inline]
    fn split_off(&mut self, n: usize) -> &'a str {
        let (ret, rest) = self.inner.split_at(n);
        self.inner = rest;
        ret
    }
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        const START_TOKEN: &str = "<|startoftext|>";
        const END_TOKEN: &str = "<|endoftext|>";

        match self.inner.as_bytes() {
            [b'\'', b's' | b't' | b'm' | b'd', ..] => Some(self.split_off(2)),
            [b'\'', c0, c1, ..] if matches!(&[*c0, *c1], b"re" | b"ve" | b"ll") => {
                Some(self.split_off(3))
            }
            [b'<', ..] if self.inner.starts_with(START_TOKEN) => {
                Some(self.split_off(START_TOKEN.len()))
            }
            [b'<', ..] if self.inner.starts_with(END_TOKEN) => {
                Some(self.split_off(END_TOKEN.len()))
            }
            _ => {
                let mut chars = self.inner.char_indices();
                match chars.next()? {
                    (i, c) if c.is_whitespace() => {
                        self.split_off(i + c.len_utf8());
                        self.next()
                    }
                    (i, c) if c.is_alphabetic() => {
                        let (i, c) = chars
                            .take_while(|(_, c)| c.is_alphabetic())
                            .last()
                            .unwrap_or((i, c));
                        Some(self.split_off(i + c.len_utf8()))
                    }
                    (i, c) => Some(self.split_off(i + c.len_utf8())),
                }
            }
        }
    }
}

pub static TABLE: [char; 256] = [
    'Ā', 'ā', 'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď', 'Đ', 'đ', 'Ē',
    'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ', 'Ġ', '!', '"', '#', '$', '%',
    '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8',
    '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^',
    '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
    'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'ġ', 'Ģ', 'ģ', 'Ĥ', 'ĥ', 'Ħ',
    'ħ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'Ĭ', 'ĭ', 'Į', 'į', 'İ', 'ı', 'Ĳ', 'ĳ', 'Ĵ', 'ĵ', 'Ķ', 'ķ', 'ĸ', 'Ĺ',
    'ĺ', 'Ļ', 'ļ', 'Ľ', 'ľ', 'Ŀ', 'ŀ', 'Ł', 'ł', '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª',
    '«', '¬', 'Ń', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½',
    '¾', '¿', 'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ð',
    'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß', 'à', 'á', 'â', 'ã',
    'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö',
    '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ',
];
