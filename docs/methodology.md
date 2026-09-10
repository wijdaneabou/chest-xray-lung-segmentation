# Méthodologie détaillée — Chest X-Ray Lung Segmentation

Ce document complète le README avec les détails techniques retirés pour garder le README principal court.

## Prétraitement

- Percentile normalization → CLAHE → resize 256×256 → normalisation Z-score.
- Statistiques de normalisation (mean/std) calculées **uniquement sur le train set**, jamais sur val/test, pour éviter toute fuite de données.

## Split

- Split stratifié sur les 4 classes diagnostiques (COVID-19, Normal, Lung Opacity, Viral Pneumonia) : 70% train / 15% validation / 15% test.
- `random_state=42` fixe, split en cascade (`train_test_split` appliqué deux fois).
- Déduplication (hash exact + perceptual hash) effectuée **avant** le split, afin d'éliminer toute fuite d'image entre les ensembles.

## Augmentation

Appliquée uniquement sur le train split :
- Géométrique : rotation, translation, scale.
- Photométrique : luminosité/contraste, correction gamma.
- Bruit : bruit gaussien, flou de mouvement.
- Flip horizontal désactivé (asymétrie thoracique réelle rendrait l'image retournée anatomiquement implausible).

## Fonction de perte et métriques

- Loss combinée : `dice_weight × DiceLoss + (1 − dice_weight) × CrossEntropy`.
- Métriques de suivi : Dice coefficient (par classe + moyenne), IoU, pixel accuracy — calculées par classe.

## Régularisation

| Technique | Rôle |
|---|---|
| BatchNorm | Stabilise l'entraînement |
| Data augmentation | Rempart principal contre le surapprentissage |
| Weight decay (Optuna) | Pénalité L2 sur l'optimiseur Adam |
| Early Stopping | Arrête l'entraînement si la métrique de validation ne s'améliore plus |
| ReduceLROnPlateau | Réduit le learning rate par paliers en cas de plateau |

## Optimisation des hyperparamètres

- Recherche Optuna (TPE), objectif : maximiser le **Validation Soft Lung Dice**.
- Hyperparamètres optimisés : learning rate, weight decay, batch size, dice weight.
- Résultat : `lr=5.94e-05`, `weight_decay=1.06e-07`, `batch_size=4`, `dice_weight=0.7`.

## Entraînement

- Optimiseur Adam, précision mixte (AMP) via `torch.autocast` + `GradScaler`.
- Checkpointing par epoch : l'état complet (poids, optimiseur, scheduler, GradScaler, historique) est sauvegardé à chaque epoch, permettant une reprise automatique en cas d'interruption (sessions Kaggle à durée limitée).
- Sauvegarde séparée du meilleur modèle selon la métrique de validation suivie.

## Explicabilité — Seg-Grad-CAM

Adaptation de Grad-CAM à la segmentation (gradient de la somme des logits d'une classe sur toute la carte spatiale), utilisée pour vérifier que le modèle s'appuie sur l'anatomie pulmonaire réelle et non sur des artefacts (lettres d'orientation, bords de l'image, équipement médical).