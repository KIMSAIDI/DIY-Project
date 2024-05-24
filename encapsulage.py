import numpy as np
import copy
from tqdm import tqdm
from activation import *
class Sequentiel:
    
    def __init__(self, modules):
        self.modules = modules
      
        
    def add_module(self, module):
        """
        Ajoute un module à la liste des modules du réseau.
        
        Args:
            module (Module): Module à ajouter.
        """
        self.modules.append(module)
        
    def forward(self, input):
        """
        Calcule la sortie du réseau pour une entrée input.
        
        Args:
            input: Entrée du réseau.
            
        Returns:
            Liste des sorties de chaque module du réseau.
        """
        liste_forwards = [input]
        for module in self.modules:
            liste_forwards.append(module.forward(liste_forwards[-1]))
        return liste_forwards[1:]
    
    def backward_delta(self, input, delta):
        """
        Calcule le gradient de la perte par rapport à l'entrée du réseau.
        
        Args:
            input : Entrée du réseau.
            delta : Gradient de la perte par rapport à la sortie du réseau.
            
        Returns:
            Liste des gradients de la perte par rapport à l'entrée de chaque module du réseau.
        """
        liste_deltas = [delta]
        for i in range(len(self.modules) - 1, 0, -1):
            self.modules[i].backward_update_gradient(input[i - 1], liste_deltas[-1])
            liste_deltas.append(self.modules[i].backward_delta(input[i - 1], liste_deltas[-1]))

        return liste_deltas
        
    
                
        
    def update_parameters(self, gradient_step):
        """ 
        Met à jour les paramètres de chaque module du réseau.
        """
        for module in self.modules:
            module.update_parameters(gradient_step)
            module.zero_grad()
        
    
    
class Optim:
    def __init__(self, net, loss, eps):
        self.net = net
        self.loss = loss
        self.eps = eps 
        
        
    def step(self, batch_x, batch_y):
        """
        Met à jour les paramètres du réseau en utilisant la descente de gradient.
        
        Args:
            batch_x : Entrée du réseau.
            batch_y : Sortie attendue du réseau.
            
        Returns:
            La valeur de la fonction de perte.                                                      
        """
        
        liste_forwards = self.net.forward(batch_x)
        batch_y_hat = liste_forwards[-1]
        loss = self.loss.forward(batch_y, batch_y_hat)
        delta = self.loss.backward(batch_y, batch_y_hat)
        list_deltas = self.net.backward_delta(liste_forwards, delta)

        self.net.update_parameters(self.eps)

        return loss
    
    
    
    def SGD(self, X_train, y_train, batch_size, epoch=100):
        """
        Entraîne le réseau en utilisant la descente de gradient stochastique.

        Args:
            X_train : Données d'entraînement.
            y_train : Labels des données d'entraînement.
            batch_size : Taille des mini-lots.
            epoch : Nombre d'itérations sur les données d'entraînement.

        Returns:
            Liste des valeurs de la fonction de perte à chaque itération.
        """
        nb_data = len(X_train)
        nb_batches = nb_data // batch_size
        if nb_data % batch_size != 0:
            nb_batches += 1

        lloss = []

        for i in tqdm(range(epoch)):
            perm = np.random.permutation(nb_data)
            X_train = X_train[perm]
            y_train = y_train[perm]

            liste_batch_x = np.array_split(X_train, nb_batches)
            liste_batch_y = np.array_split(y_train, nb_batches)
            loss_batch = 0
            for j in range(nb_batches):
                batch_x = liste_batch_x[j]
                batch_y = liste_batch_y[j]
                loss = self.step(batch_x, batch_y)
                loss_batch += loss.mean()
            
            loss_batch = loss_batch / nb_batches
            lloss.append(loss_batch)

        return lloss
    
    
    def predict(self,X_test,types):
        if types == 'softmax':
            softmax = Softmax()
            return np.argmax(softmax.forward(self.net.forward(X_test)[-1]),axis=1)
        
        elif types == 'conv':
            softmax = Softmax()
            return softmax.forward(self.net.forward(X_test)[-1])
        
        elif types == 'tanH':
            out = np.array(self.net.forward(X_test)[-1])
            return np.where(out >= 0, 1 ,0 )       
        
        elif types == 'enc':
            return self.forward(x)[0]
        elif types == 'sigmoid': 
            out = np.array(self.net.forward(X_test)[-1])
            return np.where(out >= 0.5, 1 ,0 )
    
   