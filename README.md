## ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models
### Unofficial PyTorch Implementation 

**Adding Conditional Control to Text-to-Image Diffusion Models**  
Lvmin Zhang, Anyi Rao, Maneesh Agrawala  
[https://arxiv.org/abs/2301.11093](https://arxiv.org/abs/2302.05543)

![alt text](https://github.com/faverogian/controlNet/blob/main/assets/controlNet.png?raw=true)

### Requirements
* All testing and development was conducted on 4x 16GB NVIDIA V100 GPUs
* 64-bit Python 3.8 and PyTorch 2.1 (or later). See  [https://pytorch.org](https://pytorch.org/)  for PyTorch install instructions.

For convenience, a `requirements.txt` file is included to install the required dependencies in an environment of your choice.

### Usage

A sample training script is provided which assumes a pre-trained diffusion UNet is available for use. If not, one can be trained using the utilities provided in the repository, or following the simpleDiffusion paradigm implemented in faverogian/simpleDiffusion. 

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

### Multi-GPU Training
The code is equipped with HuggingFace's [Accelerator](https://huggingface.co/docs/accelerate/en/index) wrapper for distributed training. Multi-GPU training is easily done via:
`accelerate launch --multi-gpu train.py`

### Sample Results
A model card for this project can be found at this [HuggingFace Repo](https://huggingface.co/faverogian/Smithsonian128ControlNet).

### Citations

	@misc{zhang2023addingconditionalcontroltexttoimage,
        title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
        author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
        year={2023},
        eprint={2302.05543},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2302.05543}, 
	}

    @inproceedings{Hoogeboom2023simpleDE,
	    title   = {simple diffusion: End-to-end diffusion for high resolution images},
	    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
	    year    = {2023}
	}
    
    @InProceedings{pmlr-v139-nichol21a,
	    title       = {Improved Denoising Diffusion Probabilistic Models},
	    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
	    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
	    pages       = {8162--8171},
	    year        = {2021},
	    editor      = {Meila, Marina and Zhang, Tong},
	    volume      = {139},
	    series      = {Proceedings of Machine Learning Research},
	    month       = {18--24 Jul},
	    publisher   = {PMLR},
	    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
	    url         = {https://proceedings.mlr.press/v139/nichol21a.html}
    }

    @inproceedings{Hang2023EfficientDT,
	    title   = {Efficient Diffusion Training via Min-SNR Weighting Strategy},
	    author  = {Tiankai Hang and Shuyang Gu and Chen Li and Jianmin Bao and Dong Chen and Han Hu and Xin Geng and Baining Guo},
	    year    = {2023}
	}
