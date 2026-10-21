# Petit GPT français, de zéro

Un modèle de langue de type GPT construit **entièrement de zéro** : tokenizer, architecture, pré-entraînement, évaluation, alignement par SFT, optimisation de l'inférence, puis apprentissage par renforcement (GRPO) d'un usage d'outil.

Tout a été entraîné avec des GPU gratuits (Google Colab et Kaggle, Tesla T4).

**Modèle sur Hugging Face :** [Ezeck01/petit-gpt-fr-46m](https://huggingface.co/Ezeck01/petit-gpt-fr-46m)

---

## Résultats en bref

| Étape | Résultat |
|---|---|
| Pré-entraînement | GPT de **46 M de paramètres**, 537 M de tokens de Wikipédia en français, perplexité de validation **14,9** (contre 140 pour un bigramme) |
| Grammaire | **81 %** sur 16 paires minimales en français (style BLiMP) |
| SFT | Détection de **52 % de doublons exacts** dans le jeu d'instructions ; taux de recopie sur des questions nouvelles ramené de **26 % à 16 %** pour le modèle retenu |
| Inférence | KV-cache vérifié token pour token (× 1,7) ; quantification **8 bits sans perte**, 4 bits par groupes à **÷ 7,4** en mémoire |
| GRPO agentique | Usage d'une calculatrice : **2 % → 100 %** sur la distribution d'entraînement, **92 %** sur des formulations nouvelles |

---

## Le parcours

### 0-1. Des n-grammes au tokenizer — [`scripts/`](scripts)

- `01_bigramme.py`, `02_ngramme.py` : modèles de langue par comptage sur *Les Misérables*. Plus le contexte s'allonge, plus le texte s'améliore, mais à N = 8, 75 % des contextes ne sont vus qu'une fois : le modèle récite au lieu de généraliser. C'est la limite qui motive les réseaux de neurones.
- `03_bpe.py`, `04_bpe_v2.py` : tokenizer BPE écrit à la main, avec pré-découpage en mots. Il redécouvre seul la morphologie du français (`-ement`, `-aient`, `-ité`).

### 2. Corpus et tokenizer — [`02_corpus_tokenizer.ipynb`](notebooks/02_corpus_tokenizer.ipynb)

Wikipédia en français (dump `20231101.fr`), sections finales retirées (références, liens externes…), articles courts écartés : **2 milliards de caractères**. Tokenizer BPE byte-level de **16 384 tokens** (3,73 caractères par token). Jeu de validation : 1 article sur 200, jamais vu à l'entraînement.

### 3. Le Transformer — [`03_transformer.ipynb`](notebooks/03_transformer.ipynb)

Décodeur de 12 blocs, 8 têtes d'attention, dimension 512, contexte de 512 tokens, poids partagés entre embedding et couche de sortie. Attention écrite d'abord à la main (avec le masque causal) puis avec `scaled_dot_product_attention`.

### 4. Pré-entraînement — [`04_pre_entrainement.ipynb`](notebooks/04_pre_entrainement.ipynb)

16 000 étapes de 32 768 tokens, AdamW, warmup puis cosinus, précision mixte, reprise automatique après coupure. **~4 h sur une T4.** Perte de validation : 9,79 → 2,65, sans écart avec l'entraînement (pas de sur-apprentissage).

### 5. Évaluation — [`05_evaluation.ipynb`](notebooks/05_evaluation.ipynb)

| Modèle | Contexte | Perplexité |
|---|---|---|
| Hasard | aucun | 16 384 |
| Unigramme | aucun | 1 621 |
| Bigramme | 1 token | 140 |
| **Ce modèle** | 512 tokens | **14,9** |

Paires minimales : 13/16, échecs sur le subjonctif, la 1re personne (rare dans Wikipédia) et un attracteur d'accord. QCM de connaissances : 94 % en somme des log-probabilités, mais 56 % en moyenne par token. **Le choix de la métrique change le score.**

### 6. SFT et qualité des données

- [`06a_sft.ipynb`](notebooks/06a_sft.ipynb) : ajout de deux tokens spéciaux (agrandissement de la table d'embeddings), perte masquée sur les seules réponses. Le modèle passe de « continuer un article » à « répondre puis s'arrêter ».
- [`06b_dedoublonnage.ipynb`](notebooks/06b_dedoublonnage.ipynb) : une réponse récitée mot pour mot révèle que le jeu French-Alpaca contient **51,7 % de doublons exacts** (une même blague 236 fois, 230 réponses vides). Dédoublonnage avant la séparation train/validation pour éviter la contamination.
- [`06c_controle_epoques.ipynb`](notebooks/06c_controle_epoques.ipynb) : expérience de contrôle pour séparer l'effet du dédoublonnage de celui du nombre d'étapes.

| Modèle | Perte val | Recopie (vues) | Recopie (nouvelles) | Répétition |
|---|---|---|---|---|
| Avec doublons, 2 époques | 1,114* | 27 % | 26 % | 15 % |
| Dédoublonné, 1 époque | 1,943 | 8 % | 17 % | 48 % |
| Dédoublonné, 2 époques | 1,823 | 14 % | 16 % | 28 % |
| Dédoublonné, 3 époques | **1,786** | 20 % | 15 % | 25 % |
| Dédoublonné, 4 époques | 1,816 | 24 % | 17 % | 13 % |

\* contaminée : ce modèle a vu ces exemples pendant son entraînement.

Conclusions : revoir les mêmes exemples refait monter la récitation, même sans doublons ; le dédoublonnage protège les questions nouvelles ; la perte de validation remonte à 4 époques (early stopping).

### 7. Inférence — [`07_inference.ipynb`](notebooks/07_inference.ipynb)

- **KV-cache** écrit à la main, vérifié : 511 tokens identiques sur 511. Accélération × 1,7 (limitée par le coût de lancement des opérations sur un si petit modèle).
- **Quantification** simulée, par ligne et par groupes de 64 :

| Format | Mémoire | Écart de perte |
|---|---|---|
| float32 | 186 Mo | — |
| 8 bits | 47 Mo | 0,000 |
| 4 bits, par groupes | 25 Mo | +0,062 |
| 3 bits, par groupes | 19 Mo | +0,394 |
| 2 bits, par groupes | 13 Mo | +6,894 |

### 8. GRPO agentique

Un modèle de 46 M est trop petit pour apprendre à utiliser un outil par renforcement : le GRPO ne renforce que ce qu'un modèle réussit parfois. Cette étape utilise donc **Qwen2.5-0.5B-Instruct** avec un adaptateur LoRA.

- [`08a_grpo_agentique.ipynb`](notebooks/08a_grpo_agentique.ipynb) : **GRPO implémenté à la main**. Le modèle écrit `<calc>…</calc>`, la génération s'arrête, le programme calcule et injecte le résultat, le modèle reprend. Récompense sur la seule réponse finale (l'usage de l'outil n'est pas récompensé). Les tokens injectés par l'outil sont exclus de la perte.
- [`08b_generalisation.ipynb`](notebooks/08b_generalisation.ipynb) : que vaut le 100 % ?

| Épreuve | Avant GRPO | Après GRPO |
|---|---|---|
| A. Formulations d'entraînement | 2 % | 100 % |
| B. Formulations nouvelles | 8 % | 92 % |
| C. Problèmes en deux étapes | 0 % | 68 % |
| D. Calculs faciles | 0 % | 100 % |

Ce que montre l'analyse :
- le modèle a appris **la tâche**, pas des phrases (B) ;
- il applique une recette figée, exactement **un appel** par problème, et ne décompose pas les problèmes en plusieurs étapes (C) ;
- il a **perdu des capacités générales** : avant GRPO, il répondait « Froid » au contraire de « chaud » ; après, il appelle la calculatrice sur toute question. Remèdes : mélanger les tâches, pénalité KL vers le modèle d'origine, problèmes à plusieurs étapes.

---

## Ce que j'en retiens

- **Un modèle n'est jamais meilleur que ses données** : licence Gutenberg dans le corpus, nombres supprimés par le nettoyage de Wikipédia, doublons du jeu de SFT. Chaque défaut s'est retrouvé dans les sorties.
- **Un résultat impressionnant se vérifie** : réponse trop parfaite → récitation ; 100 % → test trop facile ; résultats identiques entre deux conditions → bug de chargement.
- **Changer une seule variable à la fois** : l'expérience de contrôle de la phase 6 a renversé une première conclusion.

---

## Reproduire

Les notebooks tournent sur Google Colab (phases 2 à 5) et Kaggle (phases 6 à 8) avec un GPU T4 gratuit. Ils contiennent leurs sorties d'origine.

Dépendances principales : `torch`, `tokenizers`, `datasets`, `safetensors`, `huggingface_hub`, `transformers` et `peft` (phase 8).

## Données et modèles utilisés

- [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (CC BY-SA)
- [jpacifico/French-Alpaca-dataset-Instruct-110K](https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K) (Apache 2.0, Jonathan Pacifico)
- [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (Apache 2.0)
