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

Comparaison des architectures et encodeurs testés (métriques de test, triées par performance décroissante) :

| Rang | Modèle | Encoder | Dice coef | IoU score | Pixel accuracy |
|:---:|---|---|---:|---:|---:|
| 1 | **U-Net++** | resnet34 | **0,9890** | **0,9786** | **0,9953** |
| 2 | Attention U-Net | resnet34 | 0,9884 | 0,9774 | 0,9950 |
| 3 | U-Net++ | efficientnet-b4 | 0,9883 | 0,9773 | 0,9950 |
| 4 | Attention U-Net | efficientnet-b4 | 0,9883 | 0,9773 | 0,9950 |
| 5 | U-Net | resnet34 | 0,9881 | 0,9766 | — |
| 6 | U-Net | efficientnet-b4 | 0,9879 | 0,9761 | — |
| 7 | DeepLabV3+ | resnet34 | 0,9857 | 0,9719 | — |
| 8 | DeepLabV3+ | efficientnet-b4 | 0,9846 | 0,9698 | — |

> **Meilleur modèle : U-Net++ (resnet34)** — Dice coef = 0,9890 · IoU score = 0,9786 · Pixel accuracy = 0,9953
 
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
 
