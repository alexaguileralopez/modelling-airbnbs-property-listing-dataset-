import torch
import time
import os
import random
from datetime import datetime
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
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter



class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.data = pd.read_csv('airbnb-property-listings/listing.csv')
        self.data = clean_tabular_data(raw_dataframe= self.data)
        self.X, self.y = load_airbnb(df=self.data, label= 'Price_Night', Category= True)

        # Convert columns of self.X to float type
        self.X = self.X.astype(float)
        
        # scale the data
        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)



    def __getitem__(self, index):

        '''
        Retrieves the features and label for a specific data sample at the given index, 
        converts the features to a PyTorch tensor and returns them as a tuple
        '''

        #features = torch.tensor(self.X.iloc[index].values).float()
        features = torch.tensor(self.X[index]).float()
        label = torch.tensor(self.y.iloc[index]).float().unsqueeze(0)
       

        return(features, label)
    
    def __len__(self):
        return len(self.data)
    

class regression_NN(torch.nn.Module):

    '''Class responsible for defining the structure 
    and forward pass of the neural network.'''

    def __init__(self, hidden_layer_width, depth, dropout_rate) -> None:
        super().__init__()
        self.hyperparameters = {
            'hidden_layer_width' : hidden_layer_width,
            'depth' : depth,
            'dropout_rate' : dropout_rate
        }
        self.metrics = {}
        self.hidden_layer_width = hidden_layer_width
        self.depth = depth

        layers = []
        input_size = 11
        for i in range(depth):
            layers.append(torch.nn.Linear(input_size, hidden_layer_width))
            layers.append(torch.nn.BatchNorm1d(hidden_layer_width)) # Batch Normalisation
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate)) # Dropout
            input_size = hidden_layer_width
        
        layers.append(torch.nn.Linear(input_size, 1))
        self.layers = torch.nn.Sequential(*layers)



    def forward(self, X):
        #features = features.view(-1, 11)
        return self.layers(X)
    
    def get_hyperparameters(self):
        
        if self.hidden_layer_width is not None and self.depth is not None:
            return {
                'hidden_layer_width' : self.hidden_layer_width,
                'depth' : self.depth,
                'lr' : self.hyperparameters['lr'],
                'optimiser' : self.hyperparameters ['optimiser'],
                'dropout_rate' : self.hyperparameters['dropout_rate']
            }
    
    def calculate_rmse_loss(self,prediction, labels):
        mse_loss = F.mse_loss(prediction, labels)
        return torch.sqrt(mse_loss)
    
    def calculate_r_squared(self, prediction, labels):
        ss_tot = torch.sum((labels - torch.mean(labels))**2)
        ss_res = torch.sum((labels - prediction)**2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared
    

    def train(self, train_data_loader, val_data_loader, epochs, config):

        '''Method to train the neural network with different parameters'''

        start_time = time.time()
        optimiser_name = config['optimiser']
        lr = config['lr']

        optimiser_class = getattr(torch.optim, optimiser_name)
        optimiser = optimiser_class(self.parameters(), lr = lr) 

        #optimiser = torch.optim.SGD(params= model.parameters(), lr = 0.001) # stochastic grad descent

        writer = SummaryWriter()

        batch_idx = 0
        
        for epoch in range(epochs):
            for batch in train_data_loader:
                features, labels = batch
                prediction = self(features)
                #print(prediction)
                loss = F.mse_loss(prediction, labels)
                loss.backward() # populating gradients attribute
                optimiser.step() #optimisation step
                optimiser.zero_grad() # zeroes the gradients
                writer.add_scalar('train_loss', loss.item(), batch_idx)
                batch_idx += 1
            
            val_loss = 0.0
            val_rmse = 0.0
            num_batches = 0
            for val_batch in val_data_loader:
                val_features, val_labels = val_batch
                val_prediction = self(val_features)
                val_loss += F.mse_loss(val_prediction, val_labels).item()
                val_rmse += torch.sqrt(F.mse_loss(val_prediction, val_labels)).item()
                num_batches += 1
            val_loss /= num_batches
            val_rmse /= num_batches

            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_rmse', val_rmse, epoch)
        
        end_time = time.time()
        training_duration = end_time - start_time
        self.metrics['training_duration'] = training_duration

        self.hyperparameters['lr'] = lr
        self.hyperparameters['optimiser'] = optimiser_name
    
    def get_metrics(self, data_loader):

        '''Method to get rmse loss and r squared for a determinated data loader'''

        rmse_loss = 0.0
        r_squared = 0.0
        num_samples = 0
        total_latency = 0.0
 

        with torch.no_grad(): #disable gradient computation for evaluation
            for batch in data_loader:
                features, labels = batch
                start_time = time.time() # start time
                prediction = self(features)
                end_time = time.time() # end time


                #calculate RMSE loss
                batch_rmse = self.calculate_rmse_loss(prediction, labels)
                rmse_loss += batch_rmse.item()

                #calculate r2 
                batch_r_squared = self.calculate_r_squared(prediction, labels)
                r_squared += batch_r_squared.item()

                num_samples += features.size(0) # Count the number of samples in the batch

                # latency calculation
                batch_latency = end_time - start_time
                total_latency += batch_latency
            
            # calculate average metrics across all batches
            avg_rmse = rmse_loss / num_samples
            avg_r_squared = r_squared / num_samples

            # calculate average interference latency 
            avg_interference_latency = total_latency / num_samples
            self.metrics['interference_latency'] = avg_interference_latency

        return avg_rmse, avg_r_squared
    
    def predict(self, data_loader):

        '''Method to make predictions from a data loader'''

        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for batch in data_loader:
                features, labels = batch
                prediction = self(features)

                # Append true labels and predictions to the respective lists
                y_true_list.append(labels)
                y_pred_list.append(prediction)

        # Concatenate all batches to create the final tensors
        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)

        return y_true, y_pred
    
    def plot_results(self, data_loader):

        '''Method to plot results from a data loader'''

        y_true , y_pred = self.predict(data_loader)
        
        # Convert tensor to array and flatten
        y_true = y_true.numpy().flatten() 
        y_pred = y_pred.numpy().flatten()

        # Create a scatter plot of true labels vs. predicted labels
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_true, y=y_pred)
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("True vs. Predicted Labels")
        plt.show()

        # Create a scatter plot of residuals
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel("Predicted Labels")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()
   
    
    def evaluate_model(self, training_loader, validation_loader, testing_loader):

        '''Method to define the RMSE loss and R squared for training, validation and testing'''

        self.metrics['training_RMSE_loss'], self.metrics['training_R2'] = self.get_metrics(training_loader)
        self.metrics['validation_RMSE_loss'], self.metrics['validation_R2'] = self.get_metrics(validation_loader)
        self.metrics['testing_RMSE_loss'], self.metrics['testing_R2'] = self.get_metrics(testing_loader)

    def save_model_metrics(self, parent_folder_path):
        
        # creating folder with current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        folder_path = os.path.join(parent_folder_path, current_datetime)
        folder_counter = 1

        while True:
            folder_name = current_datetime
            if folder_counter > 1:
                folder_name += f"_v{folder_counter}"
            
            folder_path = os.path.join(parent_folder_path, folder_name)
            try:
                os.makedirs(folder_path)
                break

            except FileExistsError:
                folder_counter += 1

        
        # saving PyTorch model in a 'model.pt' file
        model_path = os.path.join(folder_path, 'model.pt')
        torch.save(self.state_dict(), model_path)

        # saving hyperparameters in a 'hyperparameters.json' file
        hyperparameters = self.get_hyperparameters()
        hyperparameters_path = os.path.join(folder_path, 'hyperparameters.json')
        with open(hyperparameters_path, 'w') as file:
            json.dump(hyperparameters, file)

        # saving metrics in a 'metrics.json' file
        metrics = self.metrics
        metrics_path = os.path.join(folder_path, 'model_metrics.json')
        with open(metrics_path, 'w') as file:
            json.dump(metrics, file)

    

            
def get_nn_config(file_path):

    ''' Function to get the neural network configuration
    '''
    with open(file_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data

def generate_nn_configs(n_configs):

    '''Function that generates a specified number of configurations 
    combining the different parameters used in this script for neural 
    network configuration'''

    optimisers = ['SGD', 'Adam']
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layer_widths = [8,16,32]
    depths = [3,4,5]
    dropout_rates = [0.2, 0.3, 0.5]

    configs = []
    for i in range(n_configs):
        config = {
            'optimiser' : random.choice(optimisers),
            'lr' : random.choice(learning_rates),
            'hidden_layer_width' : random.choice(hidden_layer_widths),
            'depth' : random.choice(depths),
            'dropout_rate' : random.choice(dropout_rates)
            
        }
        configs.append(config)

    return configs

def find_best_nn(train_loader, val_loader, test_loader, folder_path, n_configs):
    
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_metric_distance= float('-inf') # closer to 0 is best

    # generate model configurations
    configs = generate_nn_configs(n_configs= n_configs)

    for model_config in configs:

        model = regression_NN(hidden_layer_width= model_config['hidden_layer_width'], depth= model_config['depth'], dropout_rate= model_config['dropout_rate'])
        model.train(train_data_loader= train_loader, val_data_loader= val_loader, epochs= 10, config= model_config)
        model.evaluate_model(training_loader= train_loader, testing_loader= test_loader, validation_loader= val_loader)

        model.save_model_metrics(folder_path)

        # check the chosen evaluation metric to find the best model
        chosen_metric_distance = float(model.metrics['validation_R2']) # results indicated negative values for R2 in some cases
        if chosen_metric_distance > best_metric_distance:
            best_model = model
            best_metrics = model.metrics
            best_hyperparameters = model.hyperparameters
            best_metric_distance = chosen_metric_distance
    
    # Save best model in a folder
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    best_model_folder = os.path.join(folder_path, 'best_model', current_datetime)
    os.makedirs(best_model_folder)

    best_model_path = os.path.join(best_model_folder, 'best_model.pt')
    torch.save(best_model.state_dict(), best_model_path)

    with open(os.path.join(best_model_folder, 'metrics.json'), 'w') as file:
        json.dump(best_metrics, file)

    with open(os.path.join(best_model_folder, 'hyperparameters.json'), 'w') as file:
        json.dump(best_hyperparameters, file)

    return best_model, best_metrics, best_hyperparameters
    




if __name__ == '__main__':

    np.random.seed(0)
    dataset = AirbnbNightlyPriceRegressionDataset()
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)


    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=True)

    
    model_config = get_nn_config('nn_config.yaml')
    model = regression_NN(hidden_layer_width= model_config['hidden_layer_width'], depth= model_config['depth'], dropout_rate= model_config['dropout_rate'])

    model.train(train_data_loader= train_loader, val_data_loader= val_loader, epochs= 10, config= model_config)

    model.evaluate_model(training_loader= train_loader, testing_loader= test_loader, validation_loader= val_loader)

    folder_path = 'models/neural_networks/regression'
    model.save_model_metrics(folder_path)

    
    best_model, best_metrics, best_hyperparameters = find_best_nn(train_loader= train_loader, val_loader= val_loader, test_loader= test_loader, folder_path='models/neural_networks/regression', n_configs= 10)

    print('Best Metrics',best_metrics)
    print( 'Best Hyperparameters', best_hyperparameters) 
    
    #torch.save(model.state_dict(), 'regression/neural_networks/model.pt')
    
    #state_dict = torch.load('model.pt')
    
    '''
    new_model = regression_NN(hidden_layer_width= model_config['hidden_layer_width'], depth= model_config['depth'])
    new_model.load_state_dict(state_dict)
    train(new_model)'''

    
