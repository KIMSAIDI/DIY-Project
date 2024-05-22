import numpy as np
import copy
from tqdm import tqdm
class Sequentiel:
    
    def __init__(self, modules, labels=None):
        self.modules = modules
        self.labels = labels
        
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
        return liste_forwards
    
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
            liste_deltas.append(self.modules[i].backward_delta(input[i], liste_deltas[-1]))
        liste_deltas.reverse()
        return liste_deltas
    
                  
    
   
        
    def update_parameters(self, gradient_step):
        """ 
        Met à jour les paramètres de chaque module du réseau.
        """
        for module in self.modules:
            module.update_parameters(gradient_step)
            module.zero_grad()
        
    def predict(self, X):
        if self.labels is not None:
            return self.labels(self.forward(x)[0])
        return self.forward(x)[0]
 

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
    
    
    # def SGD(self, data_x, data_y, batch_size, epochs=100):
    #     """
    #     Effectue une descente de gradient stochastique (SGD).
        
    #     Parameters:
    #         data_x : np.ndarray
    #             Jeu de données.
    #         data_y : np.ndarray
    #             Labels correspondants.
    #         batch_size : int
    #             Taille des lots de données.
    #         epochs : int, optional
    #             Nombre d'époques (itérations sur l'ensemble des données). Par défaut 100.
                
    #     Returns:
    #         liste_loss : list
    #             Liste des pertes moyennes par époque.
    #     """
        
    #     # Vérifications de type et de valeur
    #     if not isinstance(data_x, np.ndarray) or not isinstance(data_y, np.ndarray):
    #         raise ValueError("data_x et data_y doivent être des tableaux numpy.")
    #     if not isinstance(batch_size, int) or batch_size <= 0:
    #         raise ValueError("batch_size doit être un entier positif.")
    #     if not isinstance(epochs, int) or epochs <= 0:
    #         raise ValueError("epochs doit être un entier positif.")
        
    #     nb_data = len(data_x)
    #     nb_batches = (nb_data + batch_size - 1) // batch_size  # Calcul du nombre de lots

    #     liste_loss = []

    #     for epoch in tqdm(range(epochs), desc="Epochs"):
    #         # Permuter les données
    #         perm = np.random.permutation(nb_data)
    #         shuffled_x = data_x[perm]
    #         shuffled_y = data_y[perm]

    #         # Effectue la descente de gradient pour chaque batch
    #         epoch_loss = 0
    #         for batch_idx in range(nb_batches):
    #             start_idx = batch_idx * batch_size
    #             end_idx = min(start_idx + batch_size, nb_data)
    #             batch_x = shuffled_x[start_idx:end_idx]
    #             batch_y = shuffled_y[start_idx:end_idx]
    #             batch_loss = self.step(batch_x, batch_y)
    #             epoch_loss += batch_loss.mean()
            
    #         epoch_loss /= nb_batches  # Moyenne de la perte pour l'époque
    #         liste_loss.append(epoch_loss)

    #     return liste_loss
        
    def SGD(self, X, Y, batch_size, epoch=10,earlystop=100):
        assert len(X) == len(Y)
        
        #shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        Y = Y[indices]
        

        #generate batch list

        batch_X  = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        batch_Y = [Y[i:i + batch_size] for i in range(0, len(Y), batch_size)]
        
        mean = []
        std = []
        minloss=float("inf")
        bestepoch = 0
        stop=0
        bestModel = self.net
        for e in range(epoch):
            tmp = []
            for x,y in zip(batch_X, batch_Y):
                tmp.append(np.asarray(self.step(x, y)).mean())
            tmp = np.asarray(tmp)
            loss = tmp.mean()
            stop+=1
            if(loss < minloss):
                stop=0
                bestepoch = e
                minloss = loss
                bestModel = copy.deepcopy(self.net)
            if stop == earlystop:
                print("early stop best epoch : ",bestepoch)
                break
            mean.append(loss)
            std.append(tmp.std())
        self.net = bestModel
        return mean, std
        
    
    def accuracy(self, y_pred, y_test):
        """
        Calcule l'exactitude des prédictions.
        
        Parameters:
            y_pred : np.ndarray
                Prédictions du modèle.
            y_test : np.ndarray
                Labels réels.
                
        Returns:
            float
                Exactitude des prédictions.
        """
        
        # Vérification des types d'entrée
        if not isinstance(y_pred, np.ndarray):
            raise ValueError("y_pred doit être un tableau numpy.")
        if not isinstance(y_test, np.ndarray):
            raise ValueError("y_test doit être un tableau numpy.")
        
        # Vérification des dimensions
        if y_pred.shape != y_test.shape:
            raise ValueError("y_pred et y_test doivent avoir la même forme.")
        
        # Calcul de l'exactitude
        return np.where(y == self.net.predict(x), 1, 0).mean()
