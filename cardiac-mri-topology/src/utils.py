import torch
import matplotlib.pyplot as plt

def plot_sample(image, ground_truth, prediction=None):
    """Visualizes an MRI slice along with its ground truth and predicted masks."""
    cols = 3 if prediction is not None else 2
    plt.figure(figsize=(5 * cols, 5))

    plt.subplot(1, cols, 1)
    plt.title("MRI Slice")
    plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')

    plt.subplot(1, cols, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(ground_truth.cpu().numpy(), cmap='jet')

    if prediction is not None:
        plt.subplot(1, cols, 3)
        plt.title("Predicted Mask")
        pred_mask = torch.argmax(prediction, dim=0)
        plt.imshow(pred_mask.cpu().numpy(), cmap='jet')

    plt.tight_layout()
    plt.show()
