#import pytorch
import numpy as np


class AnnModel:
    
    def __init__(self):
        pass

class MLPModel:
    def __init__(self, dim_in, hidden_layer_dim, dim_out, activation_type = 'Relu', reg_type = 'L2', reg = 1e-4 ):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_layers = len(hidden_layer_dim)
        self.reg_type = reg_type
        self.reg_lambda = reg
        
        #Weights, transform from input to output Dim
        self.W = []
        # Bias
        self.b = []
        # Layer output intialize
        self.H = []
        # gradients
        self.dH = []
        self.dW = []
        self.db = []

        h_dim_in = dim_in
        # hidden layer Weight Matrix
        for i in range(self.num_layers):
            #Weights, transform from input to output Dim
            self.W.append(0.01*np.random.randn(h_dim_in,hidden_layer_dim[i]))
            # Bias
            self.b.append(np.zeros( (1,hidden_layer_dim[i]) ))
            # Layer output intialize
            self.H.append(None)
            
            # gradients
            self.dH.append(None)
            self.dW.append(None)
            self.db.append(None)
            
            # Update output dim as input dim for next layer
            h_dim_in = hidden_layer_dim[i]
        
        #final layer Weight Matrix 
        self.W.append(0.01*np.random.randn(h_dim_in,dim_out))
        self.b.append(np.zeros((1,dim_out)))
        self.dW.append(None)
        self.db.append(None)
        #Output
        self.Y = None

        if(activation_type == 'Relu'):
            self.activation_fn = lambda _X: np.maximum(0,_X)
        elif(activation_type == 'tanh'):
            self.activation_fn = lambda _X: np.tanh(0,_X)
        else: # default is Relu
            self.activation_fn = lambda _X: np.maximum(0,_X)
        return # function end

    def ForwardPass(self,X):
        'Forward pass: X = input, updates self.H and self.Y'
        H = X
        for i in range(self.num_layers):
            self.H[i] = self.activation_fn( np.dot(H, self.W[i]) + self.b[i] )
            H = self.H[i]
        # final layer output
        self.Y =  (np.dot(H, self.W[i+1]) + self.b[i+1])
        return self.Y # function end
    
    def Loss(self, class_out, loss_type = 'SoftMax'):
        # compute class probability 
        num_examples = class_out.shape[0]
        exp_scores = np.exp(self.Y)
        self.probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True) 
        # compute loss
        correctc_logprob = -np.log(self.probs[range(num_examples),class_out])
        #average loss across data
        data_loss = np.sum(correctc_logprob)/num_examples

        reg_loss = 0
        reg = self.reg_lambda

        if(self.reg_type == 'L2'):
            for i in range(self.num_layers + 1):
                reg_loss += 0.5 * reg * np.sum(self.W[i] * self.W[i])
        else: # 'L1'
            for i in range(self.num_layers + 1):
                reg_loss += reg * np.sum( np.abs(self.W[i]) )

        loss = data_loss + reg_loss
        return loss

    def BackwardPass(self, X, Y_exp):
         # gradients
        num_examples = X.shape[0]
        dscores = self.probs
        dscores[range(num_examples),Y_exp] -= 1 
        dscores /= num_examples # average the delta scores across all input samples

        # back propagate
        i = self.num_layers
        self.dW[i] = np.dot(self.H[i-1].T,dscores)
        self.db[i] = np.sum(dscores,axis=0,keepdims=True)
        
        back_grad = dscores
        for i in range(self.num_layers - 1, -1, -1):
            self.dH[i] = np.dot(back_grad, self.dW[i+1].T)
           
            #backprop Relu, NOTE: BIG diff between H<=0 and H<0 
            self.dH[i][self.H[i] <= 0] = 0
           
            back_grad = self.dH[i] 
            if(i > 0):
                self.dW[i] = np.dot(self.H[i-1].T,back_grad)
                self.db[i] = np.sum(back_grad,axis=0,keepdims=True)  
            else:
                self.dW[0] = np.dot(X.T,self.dH[0])
                self.db[0] = np.sum(self.dH[0],axis=0,keepdims=True)

        # regularization gradient
        if(self.reg_type == 'L2'):
            for i in range(self.num_layers + 1):
                self.dW[i] += self.reg_lambda * self.W[i]
        else:
            for i in range(self.num_layers + 1):
                dW_l1 = np.zeros_like(self.W[i])
                dW_l1[self.W[i]<0] = -self.reg_lambda
                dW_l1[self.W[i]>0] = self.reg_lambda
                self.dW[i] += dW_l1
        return # function end
    
    def WeightUpdate(self, step_size = 1e-2):
        #Gradient decent 
        for i in range(self.num_layers + 1):
            self.W[i] += -step_size * self.dW[i]
            self.b[i] += -step_size * self.db[i]

        return # function end
    def NnAccuracy(self,X,Y_exp):
        # Training Set accuracy 
        Y = self.ForwardPass(X)
        
        preidicted_class = np.argmax(Y, axis=1)

        return np.mean(preidicted_class == Y_exp)

def main():
    pass

if __name__ == '__main__':
    main()
            





        


    

