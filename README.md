## ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models
### Unofficial PyTorch Implementation 

**Adding Conditional Control to Text-to-Image Diffusion Models**
Lvmin Zhang, Anyi Rao, Maneesh Agrawala
[https://arxiv.org/abs/2301.11093](https://arxiv.org/abs/2302.05543)

### Requirements
* All testing and development was conducted on 4x 16GB NVIDIA V100 GPUs
* 64-bit Python 3.8 and PyTorch 2.1 (or later). See  [https://pytorch.org](https://pytorch.org/)  for PyTorch install instructions.

For convenience, a `requirements.txt` file is included to install the required dependencies in an environment of your choice.

### Usage

TODO

### Multi-GPU Training
The code is equipped with HuggingFace's [Accelerator](https://huggingface.co/docs/accelerate/en/index) wrapper for distributed training. Multi-GPU training is easily done via:
`accelerate launch --multi-gpu train.py`

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
