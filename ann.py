#!/usr/bin/env python
# coding: utf-8

# In[399]:


import numpy as np



# In[400]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)

def sig_derivative(z):
    a = 1/(1+np.exp(-z))
    return a*(1-a)


def relu_derivative(x):
    if x<=0:
        return 0
    else:
        return 1
    
'''
use np.vectorize(relu_derivative)
'''


# In[ ]:





# In[401]:



class DenseL:
	#choose either relu or sigmoid
	def __init__(self, neurons, activation):
		self.activation = activation
		self.neurons = neurons

	def __str__(self):
		return "denselayer"


# In[402]:


class NNModel:
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.layers_list = []
        
    def add_layer(self, layer):
        self.layers_list.append(layer)
    
    def activations(self):
        l = []
        for layer in self.layers_list:
            l.append(layer.activation)
            
    
        
    def assign_weights_and_bias(self):
        self.W_b = {}
        neurons_in_first_layer = self.layers_list[0].neurons
        self.W_b["w"+str(0)] = np.random.randn(neurons_in_first_layer, self.input_shape[0])
        self.W_b["b"+str(0)] = np.random.randn(neurons_in_first_layer, 1)
        for i in range(1, len(self.layers_list)):
            self.W_b["w"+str(i)] = np.random.randn(self.layers_list[i].neurons, self.layers_list[i-1].neurons)
            self.W_b["b"+str(i)] = np.random.randn(self.layers_list[i].neurons, 1)
        
        
    def print_final_model(self):
        print("********************")
        print(f"input_shape->{self.input_shape}")
        for i in self.layers_list:
            print(f"{i}, neurons-> {i.neurons}, activation-> {i.activation}")
        print("**************************")
    
    def feed_forward(self, X):
        cache_z = []
        cache_a = []
        w0 = self.W_b['w0']
        b0 = self.W_b['b0']
        z0 = np.dot(w0, X) + b0
        cache_z.append(z0)
        layer_obj = self.layers_list[0]
        if layer_obj.activation == "relu":
            a0 = relu(z0)
        elif layer_obj.activation == "sigmoid":
            a0 = sigmoid(z0)
        cache_a.append(a0)
        no_of_layers = len(self.layers_list)
        temp_a = a0
        for i in range(1, no_of_layers):
            w = self.W_b['w'+str(i)]
            b = self.W_b['b'+str(i)]
            z = np.dot(w, temp_a) + b
            cache_z.append(z)
            if layer_obj.activation == "relu":
                temp_a = relu(z)
            elif layer_obj.activation == "sigmoid":
                temp_a = sigmoid(z)
            cache_a.append(temp_a)
        return temp_a, cache_z, cache_a
    
    
    def backward_propagation(self, A, y, a_cache,z_cache, X):
        dal = - (np.divide(y, A) - np.divide(1 - y, 1 - A))
        # print(f"da shape -> {dal.shape}")
        gradients = {}
        layers_len = len(self.layers_list)
        m = self.input_shape[1]
        dzl = np.multiply(dal, sig_derivative(z_cache[-1]))
        # print(f"shape of dzl -> {dzl.shape}")
        gradients["dw"+str(layers_len-1)] = (1/m)*(np.dot(dzl, np.transpose(a_cache[-2])))
        gradients["db"+str(layers_len-1)] = (1/m)*(np.sum(dzl, axis=1, keepdims=True))
        da = self.W_b["w" + str(layers_len-1)].T.dot(dzl)
        
        for i in range(layers_len-2, 0, -1):
            # print(i)
            dz = da * sig_derivative(z_cache[-(layers_len-i)])
            # print(f"shape dz {dz.shape}")
            gradients["dw"+str(i)] = (i/m)*(np.dot(dz, np.transpose(a_cache[-(layers_len-i+1)])))
            # print(f"dw shape {gradients['dw'+str(i)].shape} ")
            gradients["db"+str(i)] = (i/m)*(np.sum(dz, axis=1, keepdims=True))
            if i>0:
                da = self.W_b["w"+str(i)].T.dot(dz)
        dz = da * sig_derivative(z_cache[0])
        gradients["dw0"] = (i/m)*(np.dot(dz, np.transpose(X)))
        gradients["db0"] = (i/m)*(np.sum(dz, axis=1, keepdims=True))
            
                
        return gradients
    
    def update_weights_bias(self, gradients, lr = 0.0001):
        for i in range(len(self.layers_list)):
            self.W_b["w"+str(i)] -= lr*(gradients["dw"+str(i)])
            self.W_b["b"+str(i)] -= lr*(gradients["db"+str(i)])
             
    def cost(self, A, y):
        m = self.input_shape[1]
        return (-1/m)*np.sum(y*np.log(A)+(1-y)*np.log(1-A))
    
    def fit(self, X, y, lr = 0.001, epochs = 5):
        print("***********************************")
        for i in range(1, epochs+1):
            if i%1000 == 0:
                print(f"{i}th iteration")
            pred, z_cache, a_cache = self.feed_forward(X)
            if i%1000 == 0:
                print(pred.shape, y.shape)
                print(f"cost -> {self.cost(pred, y)}")
            gradients = self.backward_propagation(pred, y, a_cache, z_cache, X)
            self.update_weights_bias(gradients, lr = lr)


        print("done")
        
            
    
        
            
            

        
        
        
    
    
    



        


# In[403]:


model = NNModel((2, 75))
model.add_layer(DenseL(16, "sigmoid"))
model.add_layer(DenseL(16, "sigmoid"))
model.add_layer(DenseL(8, "sigmoid"))
model.add_layer(DenseL(1, "sigmoid"))
# model.print_final_model()
model.assign_weights_and_bias()


# In[404]:


import pandas as pd


# In[405]:


data = pd.read_csv("example.csv")


# In[406]:


X = data.iloc[:, 1:3]
y = data.iloc[:, -1]


# In[ ]:





# In[407]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y)


# In[408]:


x_train = x_train.T
y_train = np.array(y_train).reshape(1, 75)
model.fit(x_train, y_train, lr = 0.03, epochs = 100000)


# In[283]:


# y = np.random.rand(model.layers_list[-1].neurons, model.input_shape[1])
# re = model.backward_propagation(pred, y, a_cache, z_cache, X)





# In[ ]:




