# !pip install pytorch torchvision gdown

import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets import CelebA
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# Loads the celebA dataset from pytorch datasets, and saves only 10000 images to local dir
# This is not needed if you already have the celebA dataset in the local dir
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     # this centers the data around 0 and scales it to [-1, 1] - to make training more stable for the generator
# ])
#
# celebA = datasets.CelebA(root='data', split='all', download=True, transform=transform)
# # celebA = datasets.ImageFolder(root='data/celebA', transform=transform)
#
# # keep only 10000 random images
# celebA = torch.utils.data.Subset(celebA, torch.randperm(len(celebA))[:10000])
#
# # Split into train and test
# train_size, test_size = int(0.8 * len(celebA)), int(0.2 * len(celebA))
# train_set, test_set = torch.utils.data.random_split(celebA, [train_size, test_size])
#
# # save to local dir
# torch.save(train_set, 'data/celebA_train.pth')
# torch.save(test_set, 'data/celebA_test.pth')

# Load the celebA dataset from local dir
train_set = torch.load('data/celebA_train.pth')
test_set = torch.load('data/celebA_test.pth')

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

def unnormalize(img):
    # unnormalize (we normalized the data to [-1, 1] before)
    return img / 2 + 0.5

# Different noise functions
def gaussian_noise(image, noise_level):
    image_clone = image.clone()
    noise = torch.randn_like(image) * noise_level
    return image_clone + noise

def salt_and_pepper_noise(image, noise_level):
    image_clone = image.clone()
    noise = torch.rand_like(image_clone)
    image_clone[noise < noise_level / 2] = 0
    image_clone[noise > 1 - noise_level / 2] = 1
    return image_clone

def uniform_noise(image, noise_level):
    image_clone = image.clone()
    noise = torch.rand_like(image_clone) * noise_level
    return image_clone + noise

##########################################################
from model import Generator, Discriminator

generator = Generator(in_channels=3).to(device)
discriminator = Discriminator(input_size=128*128, in_channels=3).to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()
content_criterion = nn.L1Loss()

# Optimizers
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# for checkpointing
import os
if not os.path.exists('models'):
    os.makedirs('models')

# Train
generator.train()
discriminator.train()
num_epochs = 8
noise_level = 0.5

losses_gen, losses_disc = [], []

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(tqdm(train_loader)): # _ is the label, we don't have it here
        # Test the training:
        #if i == 1:
        #    break

        # Add noise to the real images
        # TODO: vary the noise function & noise levels
        real_images_noisy = gaussian_noise(real_images, noise_level)

        # Train the discriminator
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        fake_images = generator(real_images_noisy)
        real_disc = discriminator(real_images)
        fake_disc = discriminator(fake_images)

        real_loss = criterion(real_disc, torch.ones_like(real_disc))
        fake_loss = criterion(fake_disc, torch.zeros_like(fake_disc))
        disc_loss = (real_loss + fake_loss) / 2

        disc_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        fake_images = generator(real_images_noisy)
        fake_disc = discriminator(fake_images)

        content_loss = content_criterion(fake_images, real_images)
        gen_loss = 0.1 * criterion(fake_disc, torch.ones_like(fake_disc)) + 0.9 * content_loss

        gen_loss.backward()
        generator_optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], "
                  f"Generator loss: {gen_loss.item():.4f}, Discriminator loss: {disc_loss.item():.4f}")

        losses_gen.append(gen_loss.item())
        losses_disc.append(disc_loss.item())

    # Checkpoint the model after each epoch
    torch.save(generator.state_dict(), f'./models/generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'./models/discriminator_epoch_{epoch}.pth')

# Save the losses
np.save('losses_gen.npy', losses_gen)
np.save('losses_disc.npy', losses_disc)