import numpy as np

class MSELoss:
    """
    Fonction de perte pour la régression, la perte quadratique moyenne (MSE).
    """
    
    def forward(self, y, y_hat) :
        """
        Mesure la différence entre les valeurs prédites par le modèle y_hat 
        et les valeurs réelles y.

        Args:
            y (numpy.ndarray): valeurs réelles, taille = batch * d
            y_hat (numpy.ndarray): valeurs prédites, taille = batch * d

        Returns:
        
            numpy.ndarray : vecteur de dimension batch, chaque élément représente l'erreur moyenne au carré pour chaque exemple dans le batch
        """
        batch = y.shape[0]
        d = y.shape[1]
        errors = np.zeros(batch)  # accumulation des erreurs
        
        for i in range(batch) :
            # Calcul de l'erreur MSE pour chaque exemple et stockage dans le vecteur d'erreurs
            errors[i] = (np.sum((y[i]-y_hat[i])**2)) / d
            
        return np.array(errors)
    
    
    def backward(self, y, y_hat) :
        """
        Calcul le gradient de la fonction de perte MSE par rapport aux prédictions

        Args:
            y (numpy.ndarray): Les valeurs réelles, taille = batch * d.
            y_hat (numpy.ndarray): Les valeurs prédites, taille = batch * d.

        Returns:
            numpy.ndarray: Le gradient de la perte par rapport à y_hat, de même taille que y et y_hat.
        """
        
        return 2 * (y_hat - y) / y.shape[1] # Pour normaliser
        

class Linear:
    """
    Couche linéaire d'un réseau de neurones.
    """

    def __init__(self, input, output) :
        # Matrice des poids de la couche
        # Taille : input * output
        self.W = np.random.randn(input, output) * 0.01
        # Vecteur de biais
        # Taille : 1 * output
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
        # Mise à jour des poids
        self.W -= learning_rate * self.grad_W

        # Mise à jour des biais
        self.b -= learning_rate * self.grad_b
