import numpy as np

class MSELoss() :
    
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
        errors = np.zeros(batch)  # accumulation des erreurs
        
        for i in range(batch) :
            # Calcul de l'erreur MSE pour chaque exemple et stockage dans le vecteur d'erreurs
            errors.append(np.sum((y[i]-y_hat[i])**2))
            
        return np.array(errors)
    

class Linear(input, output) :
    
    def forward(self, matrice) :
        """
        Calcul la sortie d'une couche linéaire

        Args:
            matrice (numpy.ndarray): 
            
        Returns:
            
        """  
