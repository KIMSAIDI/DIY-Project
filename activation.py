import numpy as np
import torch
import matplotlib.pyplot as plt
from module import Module

class TanH(Module):
    def __init__(self):
        super().__init__()
        self.parameters = 0
        self.grad = 0
        
        
        
    def forward(self, batch) : 
        """
        Calcule la tangente hyperbolique de chaque élément de l'entrée batch.

        Args:
            batch (numpy.darray): données
            
        Returns: 
            numpy.darray : taille = batch.shape
        """
        
        return np.tanh(batch)
     
        
    def backward(self, batch, delta) :
        """
        Calcule le gradient de la perte par rapport à l'entrée de cette couche.

        Args:
            batch (numpy.darray): données
            delta (numpy.darray): gradient de la perte par rapport à la sortie de cette couche
            
        Returns: 
            numpy.darray : taille = batch.shape
        """
        
        return delta * (1 - np.tanh(batch)**2)
    
    def backward_update_gradient(self, grad) :
        self.grad = grad # accumule les gradients
        
    
    def update_parameters(self, learning_rate) :    
        self.parameters -= learning_rate * self.grad # met à jour les paramètres

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.parameters = 0
        self.grad = 0
        
        
    def forward(self, batch):
        """
        Calcule la sigmoïde de chaque élément de l'entrée batch.

        Args:
            batch (numpy.ndarray): données
            
        Returns: 
            numpy.ndarray : taille = batch.shape
        """
        
        return 1 / (1 + np.exp(-batch))
    
    
    def backward(self, batch, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée de cette couche.

        Args:
            batch (numpy.ndarray): données
            delta (numpy.ndarray): gradient de la perte par rapport à la sortie de cette couche
            
        Returns: 
            numpy.ndarray : taille = batch.shape
        """
        
        return delta * self.forward(batch) * (1 - self.forward(batch))
    
    
    def backward_update_gradient(self, grad) :
        self.grad = grad # accumule les gradients
        
    
    def update_parameters(self, learning_rate) :
        
        self.parameters -= learning_rate * self.grad # met à jour les paramètres
    