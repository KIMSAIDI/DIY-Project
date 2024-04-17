import numpy as np
import matplotlib.pyplot as plt
from classe_lineaire import Linear, MSELoss
from classe_non_lineaire import TanH, Sigmoid
from encapsulage import Sequentiel

# if __name__ == "__main__":

#     # Génération de données
#     np.random.seed(0) # Pour obtenir des résultats reproductibles
#     input, output, batch = 2, 1, 10
#     X = np.random.randn(batch, input)
#     w = np.array([[2.0], [-3.0]])
#     b = np.array([1.0])
#     bruit = 0.1 * np.random.randn(batch, 1)
#     Y = np.dot(X, w) + b + bruit

#     # Initalisation du modèle et la fonction de perte
#     model = Linear(input, output)
#     loss = MSELoss()


#     # Boucle d'entrainement
#     learning_rate = 1e-2
#     nb_epochs = 100
#     loss_history = [] 
#     for epoch in range(nb_epochs):
#         # Forward
#         yhat = model.forward(X) # prédiction
#         cout = loss.forward(Y, yhat) # perte
#         loss_history.append(np.mean(cout)) 

#         # Backward
#         gradient = loss.backward(Y, yhat)
#         model.backward(X, gradient) 
        
#         # Update
#         model.update(learning_rate) 
        
#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss {cout}')

#     # Afficher les poids et biais appris
#     print("\n")
#     print(f'Poids appris: {model.W.ravel()}, Biais appris: {model.b.ravel()}')


#     fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 ligne, 2 colonnes

#     # Visualisation de la perte 
#     axs[0].plot(loss_history, label='Perte MSE', color='b', lw=2, ls='-')
#     axs[0].set_xlabel('Epoch')
#     axs[0].set_ylabel('Perte')
#     axs[0].set_title('Diminution de la perte pendant l\'entraînement')
#     axs[0].legend()
#     axs[0].grid()

#     # Pour visualiser les prédictions vs les vraies valeurs
#     axs[1].scatter(Y, yhat, color='red', label='Prédictions')
#     axs[1].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, label='Correspondance')  # Ligne de correspondance parfaite
#     axs[1].set_xlabel('Vraies valeurs')
#     axs[1].set_ylabel('Prédictions')
#     axs[1].set_title('Prédictions vs vraies valeurs')
#     axs[1].legend()
#     axs[1].grid()

#     # enregistrer le modèle
#     plt.savefig("linear_regression.png")
    
    
#     # Affichage de la figure avec les deux subplots
#     plt.tight_layout()  
#     plt.show()

    
    

# if __name__ == "__main__":
#     # Générer des données aléatoires
#     X = np.random.randn(100, 5)  # 100 exemples, 5 dimensions
#     y = np.random.randn(100, 1)   # 100 exemples, 1 dimension pour les cibles

#     # Définir les dimensions d'entrée et de sortie des couches linéaires
#     input_dim = X.shape[1]
#     output_dim = 1  # Une seule sortie pour la régression

#     # Créer les couches du réseau de neurones
#     linear_layer = Linear(input_dim, output_dim)
#     tanh_layer = TanH()
#     sigmoid_layer = Sigmoid()

#     # Appliquer les différentes couches en séquence
#     linear_output = linear_layer.forward(X)
#     tanh_output = tanh_layer.forward(linear_output)
#     sigmoid_output = sigmoid_layer.forward(linear_output)

#     # Calculer la perte pour chaque sortie
#     mse_loss = MSELoss()

#     mse_loss_tanh = mse_loss.forward(y, tanh_output)
#     mse_loss_sigmoid = mse_loss.forward(y, sigmoid_output)

#     print("MSE Loss (TanH):", mse_loss_tanh)
#     print("MSE Loss (Sigmoid):", mse_loss_sigmoid)

#     # Mettre à jour les paramètres du réseau via la rétropropagation
#     learning_rate = 0.01
#     delta_tanh = mse_loss.backward(y, tanh_output)
#     delta_sigmoid = mse_loss.backward(y, sigmoid_output)

#     tanh_layer.backward_update_gradient(delta_tanh)
#     sigmoid_layer.backward_update_gradient(delta_sigmoid)

#     tanh_layer.update_parameters(learning_rate)
#     sigmoid_layer.update_parameters(learning_rate)

#     # Visualiser les résultats
#     plt.figure(figsize=(12, 6))

#     # Visualiser l'entrée
#     plt.subplot(1, 3, 1)
#     plt.scatter(X[:, 0], X[:, 1], c='b', label='Input')
#     plt.title('Entrée')
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#     plt.legend()

#     # Visualiser la sortie de TanH
#     plt.subplot(1, 3, 2)
#     plt.scatter(tanh_output[:, 0], y, c='r', label='Prediction (TanH)')
#     plt.title('Prédiction (TanH)')
#     plt.xlabel('Sortie TanH')
#     plt.ylabel('Vraie Valeur')
#     plt.legend()

#     # Visualiser la sortie de Sigmoid
#     plt.subplot(1, 3, 3)
#     plt.scatter(sigmoid_output[:, 0], y, c='g', label='Prediction (Sigmoid)')
#     plt.title('Prédiction (Sigmoid)')
#     plt.xlabel('Sortie Sigmoid')
#     plt.ylabel('Vraie Valeur')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    # dimensions des données et des couches
    input_dim = 7
    hidden_dim = 5
    output_dim = 1
    batch_size = 32

    modele = Sequentiel()

    # ajout des modules au réseau
    modele.add_module(Linear(input_dim, hidden_dim))
    modele.add_module(TanH())
    modele.add_module(Linear(hidden_dim, output_dim))
    modele.add_module(Sigmoid())

    # données d'entrée et valeurs cibles
    X = np.random.randn(batch_size, input_dim)
    y_true = np.random.randn(batch_size, output_dim)

    # (forward)
    output = modele.forward(X)

    # Calcul de la perte
    loss_func = MSELoss()
    loss = loss_func.forward(y_true, output)

    # Rétropropagation
    grad_loss = loss_func.backward(y_true, output)
    grad_input = modele.backward(X, grad_loss)

    # Mise à jour des poids
    for module in modele.modules:
        if isinstance(module, Linear):
            module.update()

    # Affichage de la perte
    print("Loss:", loss)

