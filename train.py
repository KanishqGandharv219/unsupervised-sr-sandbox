# train.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# 1. Data: CIFAR-10 with low-res -> high-res pairs
def get_dataloaders(batch_size=64, scale_factor=2):
    transform_hr = transforms.Compose([
        transforms.ToTensor()
    ])

    train_hr = datasets.CIFAR10(root="./data", train=True,
                                transform=transform_hr, download=True)
    test_hr = datasets.CIFAR10(root="./data", train=False,
                               transform=transform_hr, download=True)

    def downsample(img):
        _, h, w = img.shape
        lr = transforms.functional.resize(img, (h // scale_factor, w // scale_factor))
        lr = transforms.functional.resize(lr, (h, w))
        return lr

    class SRDataset(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, _ = self.base[idx]
            lr = downsample(img)
            hr = img
            return lr, hr

    train_ds = SRDataset(train_hr)
    test_ds = SRDataset(test_hr)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 2. Model: simple conv autoencoder-ish SR net
class SimpleSRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# 3. Training loop
def train(num_epochs=5, lr=1e-3, batch_size=64, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_dataloaders(batch_size=batch_size)

    model = SimpleSRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * lr_imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train MSE: {epoch_loss:.6f}")

    torch.save(model.state_dict(), "simple_sr.pth")
    print("Saved model to simple_sr.pth")

if __name__ == "__main__":
    train()
