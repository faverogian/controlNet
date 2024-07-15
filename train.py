"""
A script for training a ControlNet on the Smithsonian Butterflies dataset.

An adaptation of the HuggingFace ControlNet training script.
"""

from diffusion.unet import UNetCondition2D, ControlUNet
from diffusion.control_net import ControlNet
from utils.canny import AddCannyImage

from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np


class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 4
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 5e-5
    lr_warmup_steps = 10000
    save_image_epochs = 50
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm-butterflies-128"  # the model name locally and on the HF Hub
    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "faverogian/Smithsonian128ControlNet"  # the name of the repository to create on the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


def main():
    
    config = TrainingConfig

    dataset_name = "huggan/smithsonian_butterflies_subset"

    dataset = load_dataset(dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        conditions = [AddCannyImage()(image) for image in images]
        return {"images":images, "conditions":conditions}

    dataset.set_transform(transform)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    unet = UNetCondition2D.from_pretrained("faverogian/Smithsonian128UNet", variant="fp16")
    controlnet = ControlUNet.from_unet(unet, conditioning_channels=1)

    optimizer = torch.optim.Adam(controlnet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    controlnet_model = ControlNet(
        unet=unet,
        controlnet=controlnet,
        image_size=config.image_size
    )

    controlnet_model.train_loop(
        config=config,
        optimizer=optimizer,
        train_dataloader=train_loader,
        lr_scheduler=lr_scheduler
    )


if __name__ == '__main__':
    main()