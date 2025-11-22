from core.dataset import ClinicalDataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.decision_tree import DecisionTree
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from utils.processing import DataProcessor
import numpy as np


def main():
    print("=" * 60)
    print("SYSTÈME DE PRÉDICTION CLINIQUE")
    print("=" * 60)

    # 1. Charger les données
    print("\n1. Chargement des données...")
    dataset = ClinicalDataset("data/train_data.csv")
    dataset.load_data()
    X, y = dataset.split_features_target(target_column='target')
    X_train, X_test, y_train, y_test = dataset.get_train_test_split()
    print(f"   Données d'entraînement: {X_train.shape}")
    print(f"   Données de test: {X_test.shape}")

    # 2. Choisir et entraîner le modèle
    print("\n2. Entraînement du modèle...")

    # Option 1: Régression Logistique
    model = LogisticRegression(learning_rate=0.01, n_iterations=1000)

    # Option 2: Réseau de Neurones
    # model = NeuralNetwork(hidden_layers=[64, 32], learning_rate=0.01, n_iterations=1000)

    # Option 3: Arbre de Décision
    # model = DecisionTree(max_depth=5, min_samples_split=2, criterion='gini')

    trainer = Trainer(model)
    trained_model = trainer.train(X_train, y_train, normalize=True)

    # 3. Évaluer le modèle
    print("\n3. Évaluation du modèle...")
    evaluator = Evaluator(trained_model, trainer.preprocessor)
    results = evaluator.evaluate(X_test, y_test, normalize=True)
    evaluator.print_evaluation(results)

    # 4. Sauvegarder le modèle
    print("\n4. Sauvegarde du modèle...")
    trainer.save_model('models/clinical_model.pkl')

    # 5. Test de prédiction
    print("\n5. Test de prédiction sur un nouveau patient...")
    new_patient = X_test[0:1]
    prediction = trained_model.predict(new_patient)
    probability = trained_model.predict_proba(new_patient)

    print(f"   Prédiction: {'Infecté' if prediction[0] == 1 else 'Sain'}")
    print(f"   Probabilité: {probability[0]:.4f}")

    # 6. Afficher les infos du modèle (si c'est un Decision Tree)
    if isinstance(model, DecisionTree):
        print(f"\n   Profondeur de l'arbre: {model.get_tree_depth()}")
        print(f"   Nombre de feuilles: {model.count_leaves()}")

    print("\n" + "=" * 60)
    print("PROCESSUS TERMINÉ AVEC SUCCÈS!")
    print("=" * 60)


if __name__ == "__main__":
    main()