import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
from tabular_data import load_airbnb, clean_tabular_data
from sklearn.model_selection import train_test_split
from tabular_data import clean_tabular_data, load_airbnb
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter



class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        '''
        For this, instead of overcomplicating, the file that could be used instead of the raw csv could be the clean_tabular_data csv and load it into load airbnb
        '''

        self.data = pd.read_csv('airbnb-property-listings/listing.csv')
        self.data = clean_tabular_data(raw_dataframe= self.data)
        self.X, self.y = load_airbnb(df=self.data, label= 'Price_Night')

        # Convert columns of self.X to float type
        self.X = self.X.astype(float)


    def __getitem__(self, index):

        '''
        Retrieves the features and label for a specific data sample at the given index, 
        converts the features to a PyTorch tensor and returns them as a tuple
        '''

        features = torch.tensor(self.X.iloc[index].values).float()
        label = torch.tensor(self.y.iloc[index]).float().unsqueeze(0)
       

        return(features, label)
    
    def __len__(self):
        return len(self.data)
    

class regression_NN(torch.nn.Module):

    '''
    Class treating the Regression problem. 
    
    '''
    def __init__(self) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(     
            torch.nn.Linear(11,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,1),
        )

    def forward(self, X):
        #features = features.view(-1, 11)
        return self.layers(X)
    

def train(model, config, train_data_loader, val_data_loader, epochs):

    optimiser = torch.optim.SGD(params= model.parameters(), lr = 0.001) # stochastic grad descent

    writer = SummaryWriter()

    batch_idx = 0
    
    for epoch in range(epochs):
        for batch in train_data_loader:
            features, labels = batch
            prediction = model(features)
            #print(prediction)
            loss = F.mse_loss(prediction, labels)
            loss.backward() # populating gradients attribute
            optimiser.step() #optimisation step
            optimiser.zero_grad() # zeroes the gradients
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
        
        val_loss = 0.0
        val_rmse = 0.0
        num_batches = 0
        for val_batch in val_data_loader:
            val_features, val_labels = val_batch
            val_prediction = model(val_features)
            val_loss += F.mse_loss(val_prediction, val_labels).item()
            val_rmse += torch.sqrt(F.mse_loss(val_prediction, val_labels)).item()
            num_batches += 1
        val_loss /= num_batches
        val_rmse /= num_batches

        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_rmse', val_rmse, epoch)

            
def get_nn_config(file_path):

    '''
    Function to get the neural network configuration
    '''
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data


np.random.seed(0)
    
df = pd.read_csv('airbnb-property-listings/listing.csv')
df_1= clean_tabular_data(raw_dataframe= df)
dataset = load_airbnb(df= df_1, label= 'Price_Night')


dataset = AirbnbNightlyPriceRegressionDataset()

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

print("Train dataset size:", len(train_data))
features, labels = train_data[0]  # Access the first sample in the train dataset
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)


train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)


# Get the dimensions of the DataLoader
num_samples = len(train_dataloader.dataset)
batch_size = train_dataloader.batch_size
num_batches = len(train_dataloader)

print("Number of samples:", num_samples)
print("Batch size:", batch_size)
print("Number of batches:", num_batches)


model = regression_NN()
train(train_data_loader= train_dataloader, val_data_loader= val_dataloader, model= model, epochs = 10)