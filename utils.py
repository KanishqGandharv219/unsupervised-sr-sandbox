# utils.py
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from train import SimpleSRCNN, get_dataloaders

def show_examples(device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # get some test data
    _, test_loader = get_dataloaders(batch_size=4)
    lr_imgs, hr_imgs = next(iter(test_loader))

    # load model
    model = SimpleSRCNN().to(device)
    model.load_state_dict(torch.load("simple_sr.pth", map_location=device))
    model.eval()

    lr_imgs = lr_imgs.to(device)
    with torch.no_grad():
        sr_imgs = model(lr_imgs).cpu()

    lr_imgs = lr_imgs.cpu()

    fig, axs = plt.subplots(3, 4, figsize=(8, 6))
    for i in range(4):
        axs[0, i].imshow(lr_imgs[i].permute(1, 2, 0).clamp(0, 1))
        axs[0, i].set_title("LR")
        axs[0, i].axis("off")

        axs[1, i].imshow(sr_imgs[i].permute(1, 2, 0).clamp(0, 1))
        axs[1, i].set_title("SR")
        axs[1, i].axis("off")

        axs[2, i].imshow(hr_imgs[i].permute(1, 2, 0).clamp(0, 1))
        axs[2, i].set_title("HR")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_examples()
