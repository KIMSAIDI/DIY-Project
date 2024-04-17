import numpy as np

class Sequentiel:
    
    def __init__(self):
        self.modules = []
        
    def add_module(self, module):
        """
        Ajoute un module à la liste des modules du réseau.
        
        Args:
            module (Module): Module à ajouter.
        """
        self.modules.append(module)
        
    
    def forward(self, x):
        """
        Calcule la sortie du réseau pour une entrée x.
        
        Args:
            x (numpy.ndarray): Entrée du réseau.
            
        Returns:
            numpy.ndarray : Sortie du réseau.
        """
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def backward(self, x, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée du réseau.
        
        Args:
            x (numpy.ndarray): Entrée du réseau.
            delta (numpy.ndarray): Gradient de la perte par rapport à la sortie du réseau.
            
        Returns:
            numpy.ndarray : Gradient de la perte par rapport à l'entrée du réseau.
        """
        for module in reversed(self.modules):
            delta = module.backward(x, delta)
        return delta
    
    
        
        