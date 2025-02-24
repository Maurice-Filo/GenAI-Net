# Setting up the file path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Packages
from Input_Output_Rxn_Networks.IOCRN_MassAction import IOCRN_MassAction
from Input_Output_Rxn_Networks.ParameterSequenceGenerator import ParameterSequenceGenerator
import torch
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import types
import time

# Redirecting print output to a file
sys.stdout = open('Generating Parameters/Output.txt', 'w', buffering=1)

# Filename
print('- Running ', os.path.basename(__file__))

# Select Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\n- The device used is:' , device)

# Hyperparameters
N_grid = 100                            # Number of grid points for the parameter range
N_samples = 1000                        # Number of samples to generate for Monte Carlo estimations
entropy_weight = 10                     # Initial Entropy weight for the loss function
entropy_update_coefficient = 0.5        # Coefficient for updating the entropy weight
entropy_schedule = 1000                 # Number of epochs before updating the entropy weight
minimum_entropy_weight = 0.1            # Minimum entropy weight
num_epochs = 15000                      # Number of epochs to train the model
learning_rate = 0.001                   # Learning rate for the optimizer

# Construct an IOCRN modeling a Gene Expression process
stoichiometry_reactants = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]]).int().to(device)
stoichiometry_products = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]]).int().to(device)
gamma_1 = 2; gamma_2 = 2; k = 4
true_parameters = torch.tensor([gamma_1, gamma_2, k, 1]).float().to(device)
input_influence_matrix = torch.tensor([[0, 0, 0, 1]]).int().to(device)
outputs = torch.tensor([2]).int().to(device)
IOCRN_true = IOCRN_MassAction(stoichiometry_reactants, stoichiometry_products, true_parameters, input_influence_matrix, outputs)
species = ['X_1', 'X_2']
inputs = ['u']
print('\n- The reaction network is:')
IOCRN_true.print_reactions(species, inputs)

# Replace Dose response function with closed form solution
def dose_response(self, u_vec, param_vec):    
        if len(param_vec.shape) > 1:
            u_vec = u_vec.repeat( param_vec.shape[1], 1).T
        gamma_1 = param_vec[0]
        gamma_2 = param_vec[1]
        k = param_vec[2]
        c = param_vec[3]
        y_vec = k * c * u_vec / (gamma_1 * gamma_2)
        return y_vec.detach()
IOCRN_true.dose_response = types.MethodType(dose_response, IOCRN_true)

# Generate the dose response data
N_u = 100
u_Data = torch.linspace(0.1, 10, N_u).float().to(device)
y_Data = IOCRN_true.dose_response(u_Data, true_parameters)
y_Data = y_Data

# Construct the IOCRN with unknown parameters and the parameter grid
MyIOCRN = IOCRN_MassAction(stoichiometry_reactants, stoichiometry_products, [], input_influence_matrix, outputs)
MyIOCRN.dose_response = types.MethodType(dose_response, MyIOCRN)
N_parameters = 3
theta_1 = torch.linspace(1, 5, N_grid).to(device)
theta_2 = torch.linspace(1, 5, N_grid).to(device)
theta_3 = torch.linspace(1, 25, N_grid).to(device)
theta_4 = torch.linspace(1, 1, N_grid).to(device)
parameter_grid = torch.stack([theta_1, theta_2, theta_3, theta_4], dim=0).to(device)

# Construct the Parameter Sequence Generator
MyPSG = ParameterSequenceGenerator(MyIOCRN, parameter_grid, LSTM_hidden_size=128, FFNN_hidden_size=128, FFNN_num_layers=5, num_samples=N_samples).to(device)

# Train the parameter generator
History_Flag = True
manifold_dim = 0
h0 = torch.ones(MyPSG.num_samples, MyPSG.LSTM_hidden_size).float().to(device)
c0 = torch.ones(MyPSG.num_samples, MyPSG.LSTM_hidden_size).float().to(device)
optimizer = torch.optim.Adam(MyPSG.parameters(), lr=learning_rate)
loss_history = []
entropy_history = []
entropy_weight_history = []
print('\n- Training the Parameter Sequence Generator:')
tic = time.time()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss_for_each_sample, loss_mean, total_logPs, total_entropy, parameters, CRN_outputs = MyPSG.compute_loss(u_Data, y_Data, h0, c0, manifold_dim)
    entropy_mean = torch.mean(total_entropy)
    loss_for_gradient = torch.mean(loss_for_each_sample.detach() * total_logPs) - entropy_weight * entropy_mean - torch.mean(entropy_weight * total_entropy.detach() * total_logPs)
    loss_for_gradient.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch: %d, Mean Loss: %.4f, Entropy: %.4f, Entropy Weight: %.4f' % (epoch, loss_mean.item(), entropy_mean.item(), entropy_weight))
        if History_Flag:
            loss_history.append(loss_mean.item())
            entropy_history.append(entropy_mean.item())
            entropy_weight_history.append(entropy_weight)
    if epoch % entropy_schedule == 0 and epoch != 0:
        entropy_weight = max(entropy_weight * entropy_update_coefficient, minimum_entropy_weight)
toc = time.time()
print('\n- Training Time: %.2f' % (toc - tic))

# Save the model
MyPSG_Trained = {
    'model_state_dict': MyPSG.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'hyperparameters': {"learning_rate": learning_rate, 
                        "pochs": num_epochs, 
                        "entropy_weight": entropy_weight,
                        "entropy_update_coefficient": entropy_update_coefficient,
                        "entropy_schedule": entropy_schedule,
                        "minimum_entropy_weight": minimum_entropy_weight,
                        "N_grid": N_grid,
                        "N_samples": N_samples}
                        }

torch.save(MyPSG_Trained, "Generating Parameters/PSG_GeneExpression.pth")