# chest-xray-lung-segmentation

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
