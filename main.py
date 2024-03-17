import numpy as np
import matplotlib.pyplot as plt
from classe_lineaire import Linear, MSELoss

if __name__ == "__main__":

    # Génération de données
    np.random.seed(0) # Pour obtenir des résultats reproductibles
    input, output, batch = 2, 1, 10
    X = np.random.randn(batch, input)
    w = np.array([[2.0], [-3.0]])
    b = np.array([1.0])
    bruit = 0.1 * np.random.randn(batch, 1)
    Y = np.dot(X, w) + b + bruit

    # Initalisation du modèle et la fonction de perte
    model = Linear(input, output)
    loss = MSELoss()


    # Boucle d'entrainement
    learning_rate = 1e-2
    nb_epochs = 100
    loss_history = [] 
    for epoch in range(nb_epochs):
        # Forward
        yhat = model.forward(X) # prédiction
        cout = loss.forward(Y, yhat) # perte
        loss_history.append(np.mean(cout)) 

        # Backward
        gradient = loss.backward(Y, yhat)
        model.backward(X, gradient) 
        
        # Update
        model.update(learning_rate) 
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss {cout}')

    # Afficher les poids et biais appris
    print("\n")
    print(f'Poids appris: {model.W.ravel()}, Biais appris: {model.b.ravel()}')


    fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 ligne, 2 colonnes

    # Visualisation de la perte 
    axs[0].plot(loss_history, label='Perte MSE', color='b', lw=2, ls='-')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Perte')
    axs[0].set_title('Diminution de la perte pendant l\'entraînement')
    axs[0].legend()
    axs[0].grid()

    # Pour visualiser les prédictions vs les vraies valeurs
    axs[1].scatter(Y, yhat, color='red', label='Prédictions')
    axs[1].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2, label='Correspondance')  # Ligne de correspondance parfaite
    axs[1].set_xlabel('Vraies valeurs')
    axs[1].set_ylabel('Prédictions')
    axs[1].set_title('Prédictions vs vraies valeurs')
    axs[1].legend()
    axs[1].grid()

    # Affichage de la figure avec les deux subplots
    plt.tight_layout()  
    plt.show()

