#[derive(Debug)]
pub struct DeviceConfig {
    pub text_encoder: Device,
    pub vae_encoder: Device,
    pub vae_decoder: Device,
    pub unet: Device,
    pub safety_checker: Device,
}

impl DeviceConfig {
    pub fn uniform(device: Device) -> Self {
        Self {
            text_encoder: device,
            vae_encoder: device,
            vae_decoder: device,
            unet: device,
            safety_checker: device,
        }
    }

    pub fn specialized_unet(device: Device) -> Self {
        Self {
            text_encoder: Device::Cpu,
            vae_encoder: Device::Cpu,
            vae_decoder: Device::Cpu,
            unet: device,
            safety_checker: Device::Cpu,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceId(i32);

impl DeviceId {
    pub const PRIMARY: DeviceId = DeviceId(0);
}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Device {
    Cpu,
    Cuda(DeviceId),
    TensorRt(DeviceId),
    DirectMl(DeviceId),
    RocM(DeviceId),
}

impl From<Device> for ort::ExecutionProvider {
    fn from(value: Device) -> Self {
        match value {
            Device::Cpu => ort::ExecutionProvider::cpu(),
            Device::Cuda(id) => ort::ExecutionProvider::cuda().with_device_id(id.0),
            Device::TensorRt(id) => ort::ExecutionProvider::tensorrt().with_device_id(id.0),
            Device::DirectMl(id) => ort::ExecutionProvider::directml().with_device_id(id.0),
            Device::RocM(id) => ort::ExecutionProvider::rocm().with_device_id(id.0),
        }
    }
}
