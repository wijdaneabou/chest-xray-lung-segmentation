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
## 1. Business Understanding
 
- **Problème** : la segmentation manuelle des poumons sur radiographie est chronophage et sujette à variabilité inter-observateur.
- **Objectif** : localiser automatiquement le poumon droit et le poumon gauche séparément, comme étape préalable à l'analyse de pathologies pulmonaires.
- **Critère de succès** : Dice coefficient ≥ 0.90 par poumon sur le jeu de test.
- **Approche retenue** : segmentation sémantique supervisée, comparaison de plusieurs architectures de deep learning.
---

## 2. Data Understanding
- **Volume initial** : 4 classes sources (COVID, Normal, Lung_Opacity, Viral Pneumonia), ~21 165 paires image/masque au total.
- **Format** : images PNG en niveaux de gris, masques binaires (0 = fond, 255 = poumons fusionnés, sans distinction droite/gauche à l'origine)
---
 
## 3. Data Preparation

### 3.4 Split Train / Validation / Test
 
- **Répartition** : 70 % train / 15 % validation / 15 % test.
- **Stratification** : sur les 4 classes diagnostiques sources (COVID, Normal, Lung_Opacity, Viral Pneumonia), pour garantir une répartition proportionnelle dans les 3 ensembles.
- **Reproductibilité** : `random_state=42` fixe, split fait en cascade (`train_test_split` appliqué deux fois, car `sklearn` ne coupe qu'en 2 à la fois).
- **Pourquoi c'est critique ici** : les 6 architectures sont entraînées séparément (notebooks distincts) — elles doivent toutes voir **exactement** le même split pour que la comparaison finale soit valide.

Deux stratégies selon que le modèle est entraîné from scratch ou avec un encoder pré-entraîné :
 
| Cas | Normalisation | Canaux |
|---|---|---|
| Modèles from scratch (U-Net, U-Net++, Attention U-Net, DeepLabV3+, SegNet) | Z-score `(x/255 − mean) / std`, stats calculées **sur le train uniquement** | 1 (grayscale) |
| Swin-UNet (encoder pré-entraîné ImageNet) | Normalisation ImageNet standard (mean/std officiels) | 3 (grayscale dupliqué) |
 
**Point de rigueur méthodologique** : les statistiques de normalisation (mean/std) sont calculées **uniquement sur le train set**, jamais sur val/test — calculer sur l'ensemble du dataset avant le split introduirait une fuite de données (*data leakage*), le modèle aurait alors indirectement connaissance de statistiques sur les données qu'il est censé n'avoir jamais vues.

### 3.6 Data Augmentation
 
Appliquée **uniquement sur le train split**, jamais sur validation/test. Trois catégories :
 
- **Géométrique** (image + masque, même transformation) : flip horizontal, rotation (±8-10°), translation, scale (±10-20%) — masque toujours interpolé en plus-proche-voisin, `borderValue=0` (classe fond) pour les zones créées par la transformation.
- **Photométrique** (image uniquement) : variation de luminosité/contraste, correction gamma.
- **Bruit** (image uniquement) : bruit gaussien, flou de mouvement (simulant un mouvement du patient pendant l'acquisition).
**Bug corrigé en cours de projet** : le flip horizontal doit **échanger les labels 1 et 2** après la transformation. Comme droit/gauche sont définis par la position x du centroïde dans l'image, ce qui était à gauche de l'image se retrouve à droite après un flip — sans cet échange, le modèle apprenait une association erronée entre position spatiale et label de classe.
 
---



## 4. Modeling

### 4.2 Fonction de perte et métriques
 
- **Loss combinée** : `dice_weight × DiceLoss + (1 − dice_weight) × CrossEntropy`, avec `dice_weight = 0.5`.
  - La CrossEntropy apporte un gradient stable en début d'entraînement (quand les prédictions sont quasi aléatoires, le Dice seul donne des gradients peu informatifs).
  - Le Dice gère le déséquilibre de classes (le fond représente ~75-80% des pixels, les poumons le reste) — une CrossEntropy seule biaiserait l'apprentissage vers le fond.
- **Métriques de suivi** : Dice coefficient (par classe + moyenne), IoU (Intersection over Union), pixel accuracy — toutes calculées par classe pour vérifier spécifiquement le critère métier (Dice ≥ 0.90 par poumon, pas juste en moyenne globale).


### 4.3 Régularisation
 
| Technique | Rôle |
|---|---|
| **BatchNorm** (dans chaque bloc convolutif) | Stabilise l'entraînement, effet régularisant implicite |
| **Data augmentation** | Principal rempart contre le surapprentissage, particulièrement important pour les modèles from scratch (pas de pré-entraînement pour "amortir" l'initialisation aléatoire) |
| **Dropout(0.3)** | Utilisé dans le module ASPP de DeepLabV3+ |
| **Early Stopping** (`patience=25`) | Arrête l'entraînement si `val_loss` ne s'améliore plus pendant 25 epochs consécutives — évite de continuer à surapprendre après le point optimal |
| **ReduceLROnPlateau** (`factor=0.5`, `patience=4`, `cooldown=2`, `min_lr=1e-6`) | Réduit le learning rate par paliers quand `val_loss` plafonne — permet une convergence plus fine en fin d'entraînement |
 
*Remarque* : aucune pénalité L2 explicite (`weight_decay`) n'a été ajoutée à l'optimiseur Adam — piste d'amélioration possible si un surapprentissage était constaté.



### 4.4 Optimisation et entraînement
 
- **Optimiseur** : Adam, `learning_rate = 1e-4`.
- **Batch size** : 16, `epochs` max = 50.
- **Précision mixte (AMP)** : `torch.autocast` + `GradScaler` — réduit l'empreinte mémoire des activations d'environ 40-50% et accélère l'entraînement, sans changer les résultats. Décisif pour faire tenir U-Net++ (le plus lourd des modèles from scratch, du fait de ses skip pathways imbriquées qui gardent beaucoup d'activations en mémoire simultanément).
- **Checkpointing par epoch** : à la fin de **chaque** epoch (succès ou non), l'état complet (poids du modèle, optimiseur, scheduler, `GradScaler`, historique) est sauvegardé sur disque. Au relancement, l'entraînement **reprend automatiquement** à la bonne epoch au lieu de repartir de zéro — essentiel vu les sessions Kaggle à durée limitée/interruptible.
- **Sauvegarde du meilleur modèle** : séparément du checkpoint de reprise, les poids correspondant au meilleur `val_loss` observé sont sauvegardés à part (équivalent du `ModelCheckpoint(save_best_only=True)` de Keras, réimplémenté manuellement puisque PyTorch n'a pas d'équivalent natif).

  ### 4.5 Évolution du projet — deux phases
 
1. **Phase 1 — sélection d'architecture** : 6 architectures entraînées from scratch (sauf Swin-UNet, pré-entraîné) sur le dataset Montgomery/Shenzhen/Darwin combiné, chacune dans son propre notebook, comparées sur le même split. U-Net++ est ressorti comme la meilleure architecture.
2. **Phase 2 — modèle final** : U-Net++ (et DeepLabV3+ pour comparaison) réentraînés avec des encoders **pré-entraînés ImageNet** (ResNet34, EfficientNet-B4), sur le dataset **COVID-19 Radiography Database**, plus volumineux et cliniquement plus diversifié — avec un pipeline de préparation des données renforcé (déduplication par hash + NCC, vérifications de labels systématiques, alignement scheduler/checkpoint).
## 5. Evaluation
 
### 5.1 Résultats — Phase 1 (from scratch, dataset Montgomery/Shenzhen/Darwin)
 
Test Dice (poumon droit / poumon gauche) :
 
| Architecture | Dice droit | Dice gauche |
|---|---|---|
| **U-Net++** (meilleur) | **0.9825** | **0.9799** |
| Attention U-Net | 0.9818 | 0.9789 |
| U-Net | 0.9817 | 0.9791 |
| DeepLabV3+ | 0.9814 | 0.9788 |
| SegNet (le plus faible — pas de skip connections) | 0.9794 | 0.9768 |
 
Toutes les architectures atteignent l'objectif métier (Dice ≥ 0.90), les écarts entre elles restent marginaux.
 
### 5.2 Résultats — Phase 2 (encoders pré-entraînés, dataset COVID-19 Radiography Database)
 
| Modèle | Dice droit | Dice gauche |
|---|---|---|
| U-Net + ResNet34 | 0.9863 | 0.9844 |
| U-Net + EfficientNet-B4 | 0.9855 | 0.9835 |
| **U-Net++ + ResNet34** (après déduplication, meilleur modèle final) | **0.9887** | **0.9867** |
 
### 5.3 Vérifications de robustesse
 
- **Inspection des pires cas** : visualisation des images de test avec le plus faible Lung Dice, pour confirmer visuellement que les scores agrégés élevés sont crédibles et ne cachent pas un bug de métrique/label.
- **Seg-Grad-CAM** : adaptation de Grad-CAM à la segmentation (gradient de la somme des logits d'une classe sur toute la carte spatiale), pour vérifier que le modèle se base sur l'anatomie pulmonaire et non sur des artefacts (lettres imprimées L/R, bords de l'image, équipement).
- **Cartes d'erreur** : visualisation faux positifs / faux négatifs par rapport à la vérité terrain.
## 6. Deployment
 
- **Fonction d'inférence** (`predict_lung_segmentation`) : prend le chemin d'une radiographie brute, applique le même pipeline déterministe qu'à l'entraînement (percentile → CLAHE → resize → normalisation), et retourne le masque prédit + une visualisation overlay.
- **Configuration de déploiement** (`deployment_config.json`) exportée pour chaque notebook : chemins des poids, architecture, encoder, métriques de test — utilisée par le notebook de comparaison pour agréger les résultats de tous les modèles.
- **Format de sortie** : masque 3 classes (0 = fond, 1 = poumon droit, 2 = poumon gauche), rendu en overlay coloré semi-transparent.
## Structure du repo
 
```
lung-segmentation-app/
├── app/
│   ├── app.py              # Logique principale de l'application
│   ├── config.py           # Paramètres (chemins, taille image, normalisation)
│   ├── model.py            # Définition/chargement de l'architecture
│   ├── model_utils.py      # Fonctions utilitaires (inférence, overlay du masque)
│   └── preprocessing.py    # Pipeline de prétraitement (identique à l'entraînement)
├── checkpoints/            # Poids du modèle entraîné (.pth)
├── streamlit_app.py        # Point d'entrée Streamlit
└── requirements.txt
```
 
## Installation
 
```bash
git clone https://github.com/<ton-user>/lung-segmentation-app.git
cd lung-segmentation-app
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```
 
## Lancer l'application
 
```bash
streamlit run streamlit_app.py
```
 
Dataset : [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) (Kaggle).
 
## Résultats
 
Meilleur modèle : **U-Net++ avec encoder ResNet34 pré-entraîné**, sur données dédupliquées — Dice test 0.9887 (poumon droit) / 0.9867 (poumon gauche), largement au-dessus de l'objectif métier de 0.90.
 
## Limites & pistes d'amélioration
 
- **Fuite au niveau patient non totalement exclue** : le dataset ne fournit pas d'identifiant patient ; seule la fuite au niveau image (doublons/quasi-doublons) est garantie absente.
- **Pas de pénalité L2** systématique sur les modèles from scratch.
- **Flip horizontal désactivé** par choix méthodologique (asymétrie thoracique réelle) — pourrait être réévalué avec une augmentation plus légère/probabiliste.
- Pistes : validation croisée, test sur un dataset externe non vu pendant le développement, quantification/export ONNX pour le déploiement.
## Auteur
 
Wijdane Abouzaid — Projet de Fin d'Année (PFA), ENSIASD Taroudant, spécialité AI & Data Engineering.
 
