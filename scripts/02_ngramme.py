import random
from collections import defaultdict

# ---------------------------------------------------------------
# N = nombre de caractères que le modèle regarde en arrière.
# N = 1 -> ton bigramme d'avant. Essaie 3, puis 5, puis 8.
# ---------------------------------------------------------------
N = 3

with open("texte.txt", encoding="utf-8") as f:
    texte = f.read().lower()

# ---------------------------------------------------------------
# COMPTAGE : la clé n'est plus 1 caractère, mais les N précédents.
# Exemple avec N = 3 : comptes["le "]["c"] = nombre de fois
# où "c" suit "le ".
# ---------------------------------------------------------------
comptes = defaultdict(lambda: defaultdict(int))
for i in range(len(texte) - N):
    contexte = texte[i:i + N]      # les N caractères
    suivant = texte[i + N]         # celui qui vient juste après
    comptes[contexte][suivant] += 1

probas = {}
for contexte, suivants in comptes.items():
    total = sum(suivants.values())
    probas[contexte] = {s: n / total for s, n in suivants.items()}

# ---------------------------------------------------------------
# STATISTIQUES : combien de contextes différents le modèle a vus,
# et combien il pourrait en exister en théorie.
# ---------------------------------------------------------------
vocab = len(set(texte))
print(f"N = {N}")
print(f"Contextes vus dans le texte : {len(comptes):,}")
print(f"Contextes possibles en théorie (vocab^N) : {vocab ** N:,}")
vus_une_fois = sum(1 for s in comptes.values() if sum(s.values()) == 1)
print(f"Contextes vus une seule fois : {vus_une_fois:,}")

# ---------------------------------------------------------------
# GÉNÉRATION : le contexte = les N derniers caractères générés.
# ---------------------------------------------------------------
def generer(depart, n=300):
    resultat = depart
    for _ in range(n):
        contexte = resultat[-N:]
        if contexte not in probas:
            # Contexte jamais vu : le modèle est bloqué.
            resultat += " [BLOQUÉ]"
            break
        candidats = list(probas[contexte].keys())
        poids = list(probas[contexte].values())
        resultat += random.choices(candidats, weights=poids)[0]
    return resultat

# Le départ doit faire au moins N caractères.
depart = "jean valjean marcha longtemps dans la nuit. "[-N:] if N > 0 else ""
print("\nTexte généré :\n")
print(generer(depart))
