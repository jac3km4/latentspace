use std::error::Error as StdError;

use latentspace::pipelines::{F16Mode, Prompt, TextToImage};
use latentspace::schedulers::Euler;
use latentspace::{ort, Checkpoint, Device, DeviceConfig, DeviceId};

fn main() -> Result<(), Box<dyn StdError>> {
    let env = ort::Environment::builder()
        .with_log_level(ort::LoggingLevel::Verbose)
        .build()?
        .into();

    let checkpoint = Checkpoint::<F16Mode>::load(
        env,
        &DeviceConfig::specialized_unet(Device::TensorRt(DeviceId::PRIMARY)),
        // point this to a directory containg the fp16 ONNX model
        "/home/me/checkpoints/CounterfeitAnime-onnx",
    )?;

    let images = TextToImage {
        batch: vec![Prompt {
            positive: "an astronaut having a picnic".to_owned(),
            negative: "".to_owned(),
        }],
        seed: rand::random(),
        steps: 20,
        ..Default::default()
    }
    .execute(&checkpoint, &mut Euler::default())?;

    for (i, img) in images.iter().enumerate() {
        img.to_rgb8().save(format!("{i}.png"))?;
    }
    Ok(())
}
