use ndarray::Array3;

pub mod base;
pub mod clip;
pub mod lpw;

pub(crate) struct EncodedPrompts<F> {
    pub positive: Array3<F>,
    pub negative: Array3<F>,
}
