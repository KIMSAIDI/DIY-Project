import numpy as np

class Sequentiel:
    
    def __init__(self):
        self.modules = modules # [Module1, Module2, ...]
        
    def add_module(self, module):
        """
        Ajoute un module à la liste des modules du réseau.
        
        Args:
            module (Module): Module à ajouter.
        """
        self.modules.append(module)
        
    def forward_seq(self, input):
        """
        Calcule la sortie du réseau pour une entrée input.
        
        Args:
            input: Entrée du réseau.
            
        Returns:
            Liste des sorties de chaque module du réseau.
        """
        list = []
        for module in self.modules :
            list.append(module.forward(input))
        return list
    
    def backward_delta_seq(self, x, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée du réseau.
        
        Args:
            x : Entrée du réseau.
            delta : Gradient de la perte par rapport à la sortie du réseau.
            
        Returns:
            numpy.ndarray : Gradient de la perte par rapport à l'entrée du réseau.
        """
        # A FAIRE
        pass
            
    def update_parameters_seq(self, gradient_step):
        """ 
        Met à jour les paramètres de chaque module du réseau.
        """
        for module in self.modules : 
            module.update_parameters(gradient_step)
            module.zero_grad()
            
    
    
        
        