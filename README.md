# chest-xray-lung-segmentation

## About
 
Un diagnostic fiable des pathologies pulmonaires à partir de radiographies thoraciques (CXR) nécessite d'isoler précisément la zone d'intérêt, souvent parasitée par des éléments hors-poumon (côtes, artefacts, lettres d'orientation). Ce projet propose une **segmentation pulmonaire automatique bilatérale** (poumon droit / poumon gauche distingués), comme étape préalable à toute analyse de pathologie pulmonaire.
 
Plusieurs architectures de segmentation sémantique ont été comparées pour retenir la plus performante — validée par **Seg-Grad-CAM** pour confirmer qu'elle s'appuie sur l'anatomie pulmonaire réelle.
 
---
## Dataset
 
Pour ce projet, nous utilisons le [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database). Cette base de données complète contient des images de radiographies thoraciques réparties en quatre classes distinctes : COVID-19, normal, opacité pulmonaire (Lung_Opacity), et pneumonie virale. Plus précisément, le dataset comprend 3616 images de cas positifs au COVID-19, 10 192 images classées comme normales, 6012 images d'opacité pulmonaire, et 1345 images identifiées comme pneumonie virale. Cette collection étendue permet d'entraîner nos modèles de diagnostic efficacement, garantissant une performance robuste dans l'identification et la classification de ces pathologies.
 
| Classe | Nombre d'images |
|---|---|
| COVID | 3616 |
| Normal | 10192 |
| Lung_Opacity | 6012 |
| Viral Pneumonia | 1345 |
 
---
## 6. Deployment
 
- **Fonction d'inférence** (`predict_lung_segmentation`) : prend le chemin d'une radiographie brute, applique le même pipeline déterministe qu'à l'entraînement (percentile → CLAHE → resize → normalisation), et retourne le masque prédit + une visualisation overlay.
- **Configuration de déploiement** (`deployment_config.json`) exportée pour chaque notebook : chemins des poids, architecture, encoder, métriques de test — utilisée par le notebook de comparaison pour agréger les résultats de tous les modèles.
- **Format de sortie** : masque 3 classes (0 = fond, 1 = poumon droit, 2 = poumon gauche), rendu en overlay coloré semi-transparent.

## Comment exécuter le notebook
 
### Prérequis
Avant de commencer, assure-toi d'avoir un compte Kaggle ou un accès à Google Colab, ainsi que Python 3 installé si tu exécutes le notebook en local.
 
* **Étape 1 : Télécharger le dataset**
   * Télécharge le [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).
   * Le dataset doit inclure les images et les masques pour COVID-19, Lung Opacity, Normal et Viral Pneumonia.
* **Étape 2 : Configurer ton environnement**
   * Kaggle : importe le dataset dans ton compte Kaggle et utilise-le dans un nouveau notebook.
   * Google Colab : importe le dataset sur Google Drive, monte le drive dans Colab, et met à jour les chemins en conséquence.
* **Étape 3 :** Mets à jour les chemins vers le dataset dans le notebook selon ta configuration d'environnement.
* **Étape 4 :** Exécute toutes les cellules.
---



### Méthodologie
## Methodology
 
- Split stratifié : 70% train, 15% validation et 15% test.
- Suppression des doublons avant la séparation des données.
- Prétraitement : percentile normalization, CLAHE, redimensionnement à 256×256 et normalisation Z-score.
- Les statistiques de normalisation sont calculées uniquement sur le train set.
- Augmentation appliquée uniquement aux données d'entraînement (flip horizontal désactivé).
- Modèles comparés : U-Net, DeepLabV3+, Attention U-Net et U-Net++.
- Encodeurs pré-entraînés testés : ResNet34 et EfficientNet-B4.
- Fonction de perte : combinaison Dice Loss et Cross-Entropy.
- Optimisation des hyperparamètres avec Optuna TPE.
- Entraînement avec Adam, AMP, early stopping et sauvegarde du meilleur checkpoint.
- Métriques : Dice, IoU et pixel accuracy.

  ![Pipeline de la méthodologie](docs/images/pipline.png)
Le modèle est sélectionné sur la validation. Le jeu de test est conservé pour l'évaluation finale.
 
Détails complets (fonctions, checkpointing par epoch, régularisation) : voir [`docs/methodology.md`](docs/methodology.md).


## Results

Comparaison des architectures et encodeurs testés (Test Dice, argmax) :

| Modèle | Encodeur | Dice Poumon droit | Dice Poumon gauche |
|---|---|---:|---:|
| **U-Net++** | **ResNet34** | **0,9897** | **0,9882** |
| Attention U-Net | ResNet34 | 0,9895 | 0,9872 |
| U-Net++ | EfficientNet-B4 | 0,9892 | 0,9874 |
| Attention U-Net | EfficientNet-B4 | 0,9891 | 0,9875 |
| U-Net | EfficientNet-B4 | 0,9875 | 0,9858 |
| U-Net | ResNet34 | 0,9873 | 0,9859 |
| DeepLabV3+ | ResNet34 | 0,9867 | 0,9848 |
| DeepLabV3+ | EfficientNet-B4 | 0,9858 | 0,9835 |

**Meilleur modèle global : U-Net++ (ResNet34)**

Les hyperparamètres sélectionnés par Optuna pour ce modèle sont :

```text
Learning rate : 5.94e-05
Weight decay  : 1.06e-07
Batch size    : 4
Dice weight   : 0.7
```

L'objectif d'Optuna était de maximiser le **Validation Soft Lung Dice**. Les métriques finales du test sont calculées sur les masques discrets obtenus avec `argmax`.
 
## Explainability
 
Une analyse Seg-Grad-CAM est utilisée pour vérifier que le modèle porte principalement son attention sur les régions pulmonaires et non sur des artefacts tels que :
 
- les lettres d'orientation ;
- les bords de l'image ;
- les éléments externes ;
- les dispositifs médicaux.
Des cartes d'erreur et les cas présentant les plus faibles scores Dice sont également analysés.
 
## Deployment
 
Une application Streamlit permet de :
 
1. charger une radiographie thoracique ;
2. appliquer le même prétraitement que pendant l'entraînement ;
3. générer le masque pulmonaire ;
4. afficher une superposition colorée du poumon droit et du poumon gauche.
## Project Structure
 
```text
lung-segmentation-app/
├── app/
│   ├── app.py
│   ├── config.py
│   ├── model.py
│   ├── model_utils.py
│   └── preprocessing.py
├── checkpoints/
├── notebooks/
├── docs/
│   └── methodology.md
├── streamlit_app.py
├── requirements.txt
└── README.md
```
 
## Installation
 
```bash
git clone https://github.com/<your-username>/lung-segmentation-app.git
cd lung-segmentation-app
 
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
 
pip install -r requirements.txt
```
 
## Run the Application
 
```bash
streamlit run streamlit_app.py
```
 
## Limitations
 
- Les identifiants patients ne sont pas disponibles ; une fuite au niveau patient ne peut donc pas être totalement exclue.
- Certains masques de référence peuvent contenir des imprécisions ou des erreurs d'annotation.
- Les performances n'ont pas encore été vérifiées sur un dataset externe.
- Le modèle est destiné à la recherche et au prétraitement d'images, et non à une utilisation clinique autonome.

## Limites & pistes d'amélioration
 
- **Fuite au niveau patient non totalement exclue** : le dataset ne fournit pas d'identifiant patient ; seule la fuite au niveau image (doublons/quasi-doublons) est garantie absente.
- **Pas de pénalité L2** systématique sur les modèles from scratch.
- **Flip horizontal désactivé** par choix méthodologique (asymétrie thoracique réelle) — pourrait être réévalué avec une augmentation plus légère/probabiliste.
- Pistes : validation croisée, test sur un dataset externe non vu pendant le développement, quantification/export ONNX pour le déploiement.
## Auteur
 
Wijdane Abouzaid — Projet de Fin d'Année (PFA), ENSIASD Taroudant, spécialité AI & Data Engineering.
 
