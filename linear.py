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
        self._parameters = np.random.randn(input, output) * 0.01
       
        self.W = np.random.randn(input, output) * 0.01
       
        self.b = np.zeros((1, output))                  
        
        # gradient de W et b pour la mise à jour
        self.grad_W = np.zeros((input, output))
        self.grad_b = np.zeros((1, output))
    
    def forward(self, matrice) :
        """
        Calcul la sortie d'une couche linéaire

        Args:
            matrice (numpy.ndarray): Représente l'ensemble des données d'entrée, taille = batch * input
            
        Returns:
            numpy.ndarray : Représente la sortie de la couche linéaire, taille = batch * output 
        """
        
        return np.dot(matrice, self.W) + self.b

    def backward(self, x, delta):
        """
        Calcule les gradients nécessaires pour la rétropropagation à travers cette couche linéaire.

        Args:
            x (numpy.ndarray): L'entrée de la couche, taille = batch * input.
            delta (numpy.ndarray): Le gradient de la perte par rapport à la sortie de cette couche.

        Returns:
            numpy.ndarray: Le gradient de la perte par rapport à l'entrée de cette couche.
        """
        # Gradient par rapport aux poids
        grad_W = np.dot(x.T, delta)
        
        # Gradient par rapport aux biais
        grad_b = np.sum(delta, axis=0, keepdims=True)

        # Gradient par rapport à l'entrée
        grad_x = np.dot(delta, self.W.T)
        
        # Pour la mise à jour
        self.grad_W = grad_W
        self.grad_b = grad_b
        
        return grad_x
    
    def update(self, learning_rate=1e-2):
        """
        Mise à jour des poids et biais de la couche linéaire en utilisant le taux d'apprentissage.

        Args:
            learning_rate (float, optional): Taux d'apprentissage pour la mise à jour des poids. Defaults to 1e-2.
        """
        print("shape de self.W", self.W.shape)
        
        # Mise à jour des poids
        self.W -= learning_rate * self.grad_W

        # Mise à jour des biais
        self.b -= learning_rate * self.grad_b
