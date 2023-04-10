use std::error::Error as StdError;

use image::io::Reader as ImageReader;
use latentspace::pipelines::{Fp16Mode, ImagePipeline, Prompt};
use latentspace::schedulers::EulerAncestral;
use latentspace::{ort, Checkpoint, Device, DeviceConfig, DeviceId};
fn main() -> Result<(), Box<dyn StdError>> {
    tracing_subscriber::fmt::init();

    let env = ort::Environment::builder()
        .with_log_level(ort::LoggingLevel::Verbose)
        .build()?
        .into();

    let checkpoint = Checkpoint::<Fp16Mode>::load(
        env,
        &DeviceConfig::uniform(Device::TensorRt(DeviceId::PRIMARY)),
        // point this to a directory containg the fp16 ONNX model
        "/home/me/checkpoints/CounterfeitAnime-onnx",
    )?;

    let images = ImagePipeline::default()
        .with_prompt(Prompt {
            positive: "an (astronaut) having a picnic, beautiful, (((colorful))), park, blanket"
                .to_owned(),
            negative: "".to_owned(),
        })
        .with_input_image(
            ImageReader::open("/home/me/pictures/input.png")?.decode()?,
            1.,
        )
        .with_seed(rand::random())
        .execute(&checkpoint, &mut EulerAncestral::default())?;

    for (i, img) in images.iter().enumerate() {
        img.to_rgb8().save(format!("{i}.png"))?;
    }
    Ok(())
}
