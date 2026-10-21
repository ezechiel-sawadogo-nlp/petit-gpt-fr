import random
from collections import defaultdict

# ---------------------------------------------------------------
# 1. LIRE LE TEXTE
# On met tout en minuscules pour réduire le nombre de caractères
# différents (sinon "L" et "l" seraient deux choses distinctes).
# ---------------------------------------------------------------
with open("texte.txt", encoding="utf-8") as f:
    texte = f.read().lower()

print(f"Longueur du texte : {len(texte)} caractères")
print(f"Caractères différents (le 'vocabulaire') : {len(set(texte))}")

# ---------------------------------------------------------------
# 2. COMPTER LES PAIRES (bigrammes)
# zip(texte, texte[1:]) donne toutes les paires de caractères
# qui se suivent : ("b","o"), ("o","n"), ("n","j")...
# comptes["q"]["u"] = nombre de fois où "u" suit "q".
# ---------------------------------------------------------------
comptes = defaultdict(lambda: defaultdict(int))
for actuel, suivant in zip(texte, texte[1:]):
    comptes[actuel][suivant] += 1

# ---------------------------------------------------------------
# 3. TRANSFORMER LES COMPTES EN PROBABILITÉS
# Pour chaque caractère, on divise chaque compte par le total :
# la somme des probabilités des suivants fait 1.
# ---------------------------------------------------------------
probas = {}
for actuel, suivants in comptes.items():
    total = sum(suivants.values())
    probas[actuel] = {s: n / total for s, n in suivants.items()}

# Petit coup d'oeil : qu'est-ce qui suit le plus souvent "q" ?
if "q" in probas:
    top = sorted(probas["q"].items(), key=lambda x: -x[1])[:5]
    print("\nAprès 'q', les suivants les plus probables :")
    for car, p in top:
        print(f"  {repr(car)} : {p:.1%}")

# ---------------------------------------------------------------
# 4. GÉNÉRER DU TEXTE
# On part d'un caractère, on tire le suivant au hasard selon
# les probabilités, on l'ajoute, et on recommence.
# C'est EXACTEMENT ce que fait un LLM, en beaucoup plus simple.
# ---------------------------------------------------------------
def generer(depart, n=200):
    resultat = depart
    actuel = depart
    for _ in range(n):
        if actuel not in probas:
            break
        candidats = list(probas[actuel].keys())
        poids = list(probas[actuel].values())
        actuel = random.choices(candidats, weights=poids)[0]
        resultat += actuel
    return resultat

print("\nTexte généré :\n")
print(generer("l"))
