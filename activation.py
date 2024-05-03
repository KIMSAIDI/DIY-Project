import numpy as np
import torch
import matplotlib.pyplot as plt
from module import Module

class TanH(Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, batch) : 
        """
        Calcule la tangente hyperbolique de chaque élément de l'entrée batch.

        Args:
            batch (numpy.darray): données
            
        Returns: 
            numpy.darray : taille = batch.shape
        """
        
        return np.tanh(batch)
     
        
    def backward_delta(self, batch, delta) :
        """
        Calcule le gradient de la perte par rapport à l'entrée de cette couche.

        Args:
            batch (numpy.darray): données
            delta (numpy.darray): gradient de la perte par rapport à la sortie de cette couche
            
        Returns: 
            numpy.darray : taille = batch.shape
        """
        
        return delta * (1 - np.tanh(batch)**2)
    
    def update_parameters(self, gradient_step) :
        """ Pas de poids à mettre à jour car les fonctions d'activations d'ont pas de poids à optimiser"""
        pass
    
    def backward_update_gradient(self, input, delta):
        pass

   
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, batch):
        """
        Calcule la sigmoïde de chaque élément de l'entrée batch.

        Args:
            batch (numpy.ndarray): données
            
        Returns: 
            numpy.ndarray : taille = batch.shape
        """
        
        return 1 / (1 + np.exp(-batch))
    
    
    def backward_delta(self, batch, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée de cette couche.

        Args:
            batch (numpy.ndarray): données
            delta (numpy.ndarray): gradient de la perte par rapport à la sortie de cette couche
            
        Returns: 
            numpy.ndarray : taille = batch.shape
        """
        
        return delta * (np.exp(-batch) / (1 + np.exp(-batch)) ** 2)  
    
    def update_parameters(self, gradient_step) :
        """ Pas de poids à mettre à jour car les fonctions d'activations d'ont pas de poids à optimiser"""
        pass
    
    def backward_update_gradient(self, input, delta):
        pass
   