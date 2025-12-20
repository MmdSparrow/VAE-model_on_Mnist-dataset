import os
import torch
import torchvision.transforms as transforms

from torch.optim import Adam
from utils.loss import KLD_loss
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.models_1 import Encoder, Decoder, VAEModel
from configs.common_configs import DATASET_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, OUTPUT_DIRECTORY

class Train:
    def __init__(self):
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.x_dim  = 784
            self.hidden_dim = 400
            self.latent_dim = 200
            self.mnist_transform = transforms.Compose([transforms.ToTensor()])
            self.data_loader_kwargs = {'num_workers': 1, 'pin_memory': True}
            
            self.encoder = Encoder(input_dim=self.x_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
            self.decoder = Decoder(latent_dim=self.latent_dim, hidden_dim = self.hidden_dim, output_dim = self.x_dim)
            self.model = VAEModel(Encoder=self.encoder, Decoder=self.decoder, device=self.DEVICE).to(self.DEVICE)
            

    def train_model(self):
            train_dataset = MNIST(DATASET_PATH, transform=self.mnist_transform, train=True, download=False)
            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, **self.data_loader_kwargs)
            

            
            loss_function= KLD_loss()
            optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
            self.model.train()
            for epoch in range(EPOCHS):
                    overall_loss = 0
                    for batch_idx, (x, _) in enumerate(train_loader):
                            x = x.view(BATCH_SIZE, self.x_dim)
                            x = x.to(self.DEVICE)
                            optimizer.zero_grad()
                            x_hat, mean, log_var = self.model(x)
                            loss = loss_function(x, x_hat, mean, log_var)
                            overall_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*BATCH_SIZE))
    
    
    def generate_image_from_noise(self):
        with torch.no_grad():
            noise = torch.randn(BATCH_SIZE, self.latent_dim).to(self.DEVICE)
            generated_images = self.decoder(noise)
            save_image(generated_images.view(BATCH_SIZE, 1, 28, 28), os.path.join(OUTPUT_DIRECTORY, 'generated_sample.png'))
