from enum import IntEnum

from torchvision.utils import save_image
from tqdm.auto import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from gan_test.mnist_gan.discriminators import SimpleDiscriminator, Conv1dDiscriminator
from gan_test.mnist_gan.generators import SimpleGenerator, Conv1dGenerator
from gan_test.mnist_gan.mnist_data import get_mnist


class NetworkType(IntEnum):
    SIMPLE = 0
    CONV = 1


class MnistTrainer:
    IMAGE_SIZE = 28 * 28
    GENERATOR_INPUT_SIZE = 100
    FEATURES_D = 2
    FEATURES_G = 2

    def __init__(self, num_epochs: int = 5, batch_size: int = 32, network_type: NetworkType = NetworkType.SIMPLE):
        if network_type == NetworkType.SIMPLE:
            self.generator = SimpleGenerator(self.GENERATOR_INPUT_SIZE, self.IMAGE_SIZE)
            self.discriminator = SimpleDiscriminator(self.IMAGE_SIZE)
        elif network_type == NetworkType.CONV:
            self.generator = Conv1dGenerator(noise_size=self.GENERATOR_INPUT_SIZE, channels_img=1, features_g=self.FEATURES_G)
            self.discriminator = Conv1dDiscriminator(channels_img=1, features_d=self.FEATURES_D)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def train(self):
        train_set, test_set = get_mnist()
        dataloader_train = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        for epoch in range(1, self.num_epochs + 1):
            disc_losses = []
            gen_losses = []
            for xs, _ in tqdm(dataloader_train):
                disc_loss = self.train_discriminator(xs)
                disc_losses.append(disc_loss)

                gen_loss = self.train_generator()
                gen_losses.append(gen_loss)

            tqdm.write(f"[{epoch}/{self.num_epochs}]: "
                       f"disc_loss: {torch.mean(torch.FloatTensor(disc_losses)):.4f}, "
                       f"gen_loss: {torch.mean(torch.FloatTensor(gen_losses)):.4f}")
            if epoch % 25 == 0:
                self.generate_images(num_images=100, epoch=epoch)
                tqdm.write(f"Generated images for epoch {epoch}")

    def train_discriminator(self, train_data: Tensor):

        self.discriminator.zero_grad()

        x_real = train_data
        x_real = torch.unsqueeze(x_real, dim=1)
        real_output = self.discriminator(x_real).reshape(-1)
        real_loss = self.discriminator.loss(real_output, torch.ones_like(real_output) - 0.1)

        random_noise = torch.randn((train_data.size(0), self.GENERATOR_INPUT_SIZE))
        if isinstance(self.generator, Conv1dGenerator):
            random_noise = torch.unsqueeze(random_noise, dim=-1)
        x_fake = self.generator(random_noise)

        fake_output = self.discriminator(x_fake.detach()).reshape(-1)
        fake_loss = self.discriminator.loss(fake_output, torch.zeros_like(fake_output) + 0.1)

        total_loss = (real_loss + fake_loss) / 2
        total_loss.backward()
        self.discriminator.optimizer.step()

        return total_loss.item()

    def train_generator(self):
        self.generator.zero_grad()

        random_noise = torch.randn((self.batch_size, self.GENERATOR_INPUT_SIZE))
        if isinstance(self.generator, Conv1dGenerator):
            random_noise = torch.unsqueeze(random_noise, dim=-1)

        generator_output = self.generator(random_noise)
        discriminator_output = self.discriminator(generator_output).reshape(-1)

        generator_loss = self.generator.loss(discriminator_output, torch.ones_like(discriminator_output) - 0.1)
        generator_loss.backward()
        self.generator.optimizer.step()

        return generator_loss.item()

    def generate_images(self, num_images: int, epoch: int):
        random_noise = torch.randn(num_images, self.GENERATOR_INPUT_SIZE)
        gen_output = self.generator(random_noise)

        save_image(tensor=gen_output.view(gen_output.size(0), 1, 28, 28), fp=f"./samples/epoch{epoch:03d}.png")


if __name__ == "__main__":
    trainer = MnistTrainer(num_epochs=200, batch_size=128, network_type=NetworkType.CONV)
    trainer.train()
