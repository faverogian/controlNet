"""
A script for training a ControlNet on the Smithsonian Butterflies dataset.

An adaptation of the HuggingFace ControlNet training script.
"""

from diffusion.control_net import ControlNet
from nets.unet import UNetCondition2D, ControlUNet
from utils.canny import AddCannyImage
from utils.plotter import side_by_side_plot

from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup

class TrainingConfig:
    # Optimization parameters
    learning_rate = 5e-5
    lr_warmup_steps = 10000
    train_batch_size = 64
    gradient_accumulation_steps = 1
    ema_beta = 0.9999
    ema_warmup = 500
    ema_update_freq = 10

    # Experiment parameters
    resume = True # whether to resume training from a checkpoint
    num_epochs = 750 # the number of training epochs
    save_image_epochs = 50 # how often to save generated images
    evaluation_batches = 3 # the number of batches to use for evaluation
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    experiment_path = "" # codebase root directory
    
    # Model parameters
    image_size = 128  # the generated image resolution
    backbone = "unet"  # the backbone model to use, either 'unet' or 'uvit'

    # Diffusion parameters
    pred_param = "v" # 'v', 'eps'
    schedule = "shifted_cosine" # shifted_cosine, cosine, shifted_cosine_interpolated
    noise_d = 64 # base noise dimension to shift to
    sampling_steps = 256 # number of steps to sample with

    # Seed
    seed = 0

def main():
    
    config = TrainingConfig

    # Load the dataset
    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(dataset_name, split="train")

    # Split the dataset into 80% train and 20% test
    train_test_split = dataset.train_test_split(test_size=0.2)

    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

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

    train_dataset.set_transform(transform)
    test_dataset.set_transform(transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
    )

    unet = UNetCondition2D(
        sample_size=config.image_size,  # the target image resolution
        in_channels=12,  # the number of input channels, 3 for RGB images
        out_channels=12,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128,256,512,768),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        mid_block_type="UNetMidBlock2D",
    )

    controlnet = ControlUNet.from_unet(unet, conditioning_channels=4)

    # Put both the UNet and ControlNet parameters in the same optimizer
    optimizer = torch.optim.Adam((list(controlnet.parameters()) + list(unet.parameters())), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
    )

    controlnet_model = ControlNet(
        unet=unet,
        controlnet=controlnet,
        config=config,
    )

    controlnet_model.train_loop(
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        lr_scheduler=lr_scheduler,
        plot_function=side_by_side_plot,
    )


if __name__ == '__main__':
    main()