import numpy as np
import matplotlib.pyplot as plt
import DataSource.GetData as GetData
import ANNModels.AnnModel as AnnModel
import torch

# define the data set size.
N = 100 # Number of training samples
D = 2   # Dimensions
K = 3   # Number of classes 
step_size = 1e-0
reg_type = 'L1'
reg = 1e-4 # regularization weight
## Defining Neural Network
h = [10,50] # Number of hidden neurons
   
# Generate data
spiral_data = GetData.SpiralDataGen(dim = D, num_samples = N, num_classes = K)
spiral_data.GetData(disp=False)
# 
X = spiral_data.data
y = spiral_data.class_type

##  Defining ANN MLP 
mlp_model = AnnModel.MLPModel(dim_in = D, hidden_layer_dim = h, dim_out = K, 
                             reg_type = reg_type, reg = reg, prune=True, prune_thr = 1e-5 )


'''mlp_model = AnnModel.TorchMLPModel(dim_in = D, hidden_layer_dim = h, dim_out = K)'''

for i in range(10000):
    
    # forward pass
    Y_out = mlp_model.ForwardPass(X)
    
    #Compute Loss
    loss = mlp_model.Loss(Y_out, y)
    
    mlp_model.BackwardPass(X,y)
    
    mlp_model.WeightUpdate(step_size=step_size)
    if i%1000 == 0:
        print('itr %d: loss %f' % (i,loss.item()))
        #step_size = step_size/2

precission = mlp_model.AnnAccuracy(X,y)
print("Training Accuracy %f" % (precission) )

plt = mlp_model.AnnVisualize(X,y)
plt.show()
