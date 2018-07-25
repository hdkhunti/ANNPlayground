import pandas
import numpy as np
import matplotlib.pyplot as plt
class DataSource():

    def __init__(self):
        pass
    def __del__(self):
        pass
    def GetData(self, num_samples = 0, num_classes = 0):
        pass
#end DataSource def

class SpiralDataGen(DataSource):
    
    def __init__(self, dim=2, num_samples = 0, num_classes = 0):
        super(SpiralDataGen,self).__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.dim = dim

    def __del__(self):
        super(SpiralDataGen,self).__del__()

    def GetData(self, dim = None, num_samples = None, num_classes = None, disp = False):
        # init class data
        if(dim != None):
            self.dim = dim
        if(num_samples != None):
            self.num_samples = num_samples
        if(num_classes != None):
            self.num_classes = num_classes
        
        self.data = np.zeros((self.num_samples*self.num_classes,self.dim)) # Data matrix
        self.class_type = np.zeros((self.num_samples*self.num_classes),dtype='uint8') # Class type vector

        for j in range(self.num_classes):
            #print(j)
            ix = range(self.num_samples*j,self.num_samples*(j+1)) # index 
            r  = np.linspace(0.0, 1, num=self.num_samples) # radius
            t  = np.linspace(4*j, 4*(j+1), num=self.num_samples) + np.random.randn(self.num_samples)*0.2 # theta, assign one third of 360 degrees to each class
            self.data[ix] = np.c_[ r*np.sin(t) , r*np.cos(t)] # the spiral 
            self.class_type[ix] = j  # the class type
        
        #visualize the data 
        if(disp == True):
            plt.scatter(self.data[:,0],self.data[:,1],c=self.class_type,s=40,cmap=plt.cm.Spectral)
            plt.show()
        
def main():
    # add test routine
    pass

if __name__ == '__main__':
    main()

    

    
