import numpy as np
from module import Module

class Linear(Module):
    """
    Couche linéaire d'un réseau de neurones.
    """

    def __init__(self, input, output) :
        super().__init__()      
        self.input = input
        self.output = output
        self._parameters = 2 * (np.random.rand(self.input, self.output) - 0.5) 
        self.biais = 2 * (np.random.rand(1, self.output) - 0.5)
        self.zero_grad()
    
    
    def zero_grad(self): 
        """
        Pour réinitialiser les gradients des poids et des biais.
        """
        self._gradient = np.zeros((self.input, self.output))
        self._gradient_biais = np.zeros((1, self.output))
    
    def forward(self, X) :
        """
        Calcul la sortie d'une couche linéaire

        Args:
            X (numpy.ndarray): Représente l'ensemble des données d'entrée, taille = batch * input
            
        Returns:
            numpy.ndarray : Représente la sortie de la couche linéaire, taille = batch * output 
        """
        assert X.shape[1] == self.input, ValueError("La taille de l'entrée doit être égale à la taille de l'entrée de la couche linéaire")
        return np.dot(X, self._parameters) + self.biais
    
    def update_parameters(self, gradient_step=1e-3):
        """
        Mise à jour des poids et biais de la couche linéaire en utilisant le taux d'apprentissage.

        Args:
            gradient_step : scalaire, pas du gradient
        """
        self._parameters -= gradient_step * self._gradient
        self.biais -= gradient_step * self._gradient_biais


    def backward_update_gradient(self, input, delta):
        """
        Calcule les gradients nécessaires pour la rétropropagation à travers cette couche linéaire.

        Args:
            input : L'entrée de la couche, taille = batch * self.input
            delta : Le gradient de la perte par rapport à la sortie de cette couche, taille = batch * self.output

        Returns:
            gradient des paramètres : taille = self.input * self.output
            gradient des biais : taille = 1 * self.output
        """
        assert delta.shape[1] == self.output, ValueError("La taille de delta doit être égale à la taille de sortie de la couche linéaire")
        assert input.shape[1] == self.input, ValueError("La taille de l'entrée doit être égale à la taille de l'entrée de la couche linéaire")
        
        self._gradient += np.dot(input.T, delta) 
        self._gradient_biais += np.sum(delta, axis=0, keepdims =True)
        

    def backward_delta(self, input, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée 

        Args:
            input : taille = batch * self.input
            delta : taille = batch * self.output
            
        Returns:
            dérivée de l'erreur : taille = batch * self.input
        """
        assert delta.shape[1] == self.output, ValueError("La taille de delta doit être égale à la taille de sortie de la couche linéaire")
        assert input.shape[1] == self.input, ValueError("La taille de l'entrée doit être égale à la taille de l'entrée de la couche linéaire")
        return np.dot(delta, self._parameters.T)
    
   