import numpy as np
import matplotlib.pyplot as plt
import DataSource.GetData as GetData
import ANNModels.AnnModel as AnnModel

# define the data set size.
N = 100 # Number of training samples
D = 2   # Dimensions
K = 3   # Number of classes 
reg = 1e-4 # regularization weight
step_size = 1e-0
reg_type = 'L2'

# Generate data
spiral_data = GetData.SpiralDataGen(dim = D, num_samples = N, num_classes = K)
spiral_data.GetData(disp=False)

# 
X = spiral_data.data
y = spiral_data.class_type

##  Defining ANN MLP 
mlp_model = AnnModel.MLPModel(dim_in = D, hidden_layer_dim = [100], dim_out=K, reg_type='L2', reg = 0 )

for i in range(10000):
    # forward pass
    mlp_model.ForwardPass(X)
    #Compute Loss
    loss = mlp_model.Loss(y)
    if i%1000 == 0:
        print('itr %d: loss %f' % (i,loss))
    
    mlp_model.BackwardPass(X,y)
    
    mlp_model.WeightUpdate(step_size=1e-2)

precission = mlp_model.NnAccuracy(X,y)
print("Training Accuracy %f" % (precission) )

if(True):
    ## Defining Neural Network
    h = 100 # Number of hidden neurons
    W = 0.01*np.random.randn(D,h)
    b = np.zeros((1,h))
    W2 = 0.01*np.random.randn(h,K)
    b2 = np.zeros((1,K))
    num_examples = X.shape[0] 

    for i in range(10000):
        # forward pass 
        H = np.maximum(0, np.dot(X,W) + b) # hidden layer output
        F = np.dot(H,W2) + b2 # final layer score 

        # compute class probability 
        exp_scores = np.exp(F)
        probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True) 

        # compute loss
        correctc_logprob = -np.log(probs[range(num_examples),y])
        #average loss across data
        data_loss = np.sum(correctc_logprob)/num_examples
        reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
        loss = data_loss + reg_loss

        if i%1000 == 0:
            print('Itr %d: loss %f'%(i,loss))
        
        # gradients
        dscores = probs
        dscores[range(num_examples),y] -= 1 
        dscores /= num_examples

        # back propagate
        dW2 = np.dot(H.T,dscores)
        db2 = np.sum(dscores,axis=0,keepdims=True)

        #next back prop into hidden layer
        dH = np.dot(dscores,W2.T)

        #backprop Relu, NOTE: BIG diff between H<=0 and H<0 
        dH[H <= 0] = 0

        # W delta
        dW = np.dot(X.T, dH)
        db = np.sum(dH,axis=0,keepdims=True)

        # regularization gradient
        if(reg_type == 'L2'):
            dW2 += reg * W2
            dW  += reg * W
        else:
            dW2_l1 = np.zeros_like(W2)
            dW2_l1[W2<0] = -reg
            dW2_l1[W2>0] = reg
            dW2 +=dW2_l1

            dW_l1 = np.zeros_like(W)
            dW_l1[W<0] = -reg
            dW_l1[W>0] = reg
            dW += dW_l1

        #Gradient decent 
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

    # Training Set accuracy 
    H = np.maximum(0,np.dot(X,W)+b) # hidden layer output
    F = np.dot(H,W2)+b2 # final layer score 

    preidicted_class = np.argmax(F,axis=1)
    print("Training Accuracy %f"%(np.mean(preidicted_class==y)))

    ''' 
    # initialize parameters randomly
    h = 100 # size of hidden layer
    W = 0.01 * np.random.randn(D,h)
    b = np.zeros((1,h))
    W2 = 0.01 * np.random.randn(h,K)
    b2 = np.zeros((1,K))

    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3 # regularization strength

    # gradient descent loop
    num_examples = X.shape[0]
    for i in range(10000):
    
    # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
    
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))
    
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    
    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
    
    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W
    
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

    '''
    # Visualize

    # plot the resulting classifier
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(W)
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(W2)
    plt.colorbar()

    plt.show()
    #fig.savefig('spiral_net.png')
    np.set_printoptions(precision=3,suppress=True)
    print(W)
    print(W2)
