import os
import matplotlib.pyplot as plt
from utils.wavelet import wavelet_enc_2

def side_by_side_plot(
        output_dir: str, 
        samples: list,
        batches: list, 
        epoch: int, 
        process_idx: int
    ): 
    """
    Plot the Canny and generated images side by side

    Args:
        output_dir (str): The output directory to save the plots
        samples (list): The samples to plot
        epoch (int): The epoch number
        process_idx (int): The process index

    Returns:
        None
    """

    for i, (sample, batch) in enumerate(zip(samples, batches)):

        for j in range(1):

            sample_item = sample[j] * 0.5 + 0.5
            batch_item = batch["conditions"][j] * 0.5 + 0.5

            pred = sample_item.cpu().detach().numpy().transpose(1, 2, 0)
            canny = batch_item.cpu().detach().numpy().transpose(1, 2, 0)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            ax[0].imshow(canny, cmap="gray")
            ax[0].axis("off")
            ax[0].set_title("Canny")

            ax[1].imshow(pred)
            ax[1].axis("off")
            ax[1].set_title("Generated")

            plt.tight_layout()

            # Make sure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(
                f"{output_dir}/sample_{i}_epoch_{epoch}_process_{process_idx}.png"
            )
            plt.close()

    return None
