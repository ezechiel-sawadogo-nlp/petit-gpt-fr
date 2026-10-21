import re
import time
from collections import Counter

# ---------------------------------------------------------------
# 0. LIRE ET NETTOYER (tout le texte cette fois, pas 200 000 car.)
# ---------------------------------------------------------------
with open("texte.txt", encoding="utf-8") as f:
    texte = f.read()

debut = texte.find("*** START OF")
fin = texte.find("*** END OF")
if debut != -1 and fin != -1:
    texte = texte[texte.find("\n", debut) + 1 : fin]
texte = texte.lower()

# ---------------------------------------------------------------
# 1. PRÉ-DÉCOUPAGE EN MOTS (nouveau !)
# Avant le BPE, on coupe le texte en morceaux : un mot avec
# l'espace qui le précède, ou un signe de ponctuation.
#   " l'évêque regardait." -> [" l", "'", "évêque", " regardait", "."]
# Le BPE ne fusionnera qu'À L'INTÉRIEUR d'un morceau :
# plus de tokens "dans la " à cheval sur deux mots.
# C'est le même principe que le tokenizer de GPT.
# ---------------------------------------------------------------
motif = re.compile(r" ?\w+| ?[^\s\w]+|\s+")
morceaux = motif.findall(texte)

# ---------------------------------------------------------------
# 2. L'ASTUCE QUI REND LE BPE RAPIDE
# "jean" apparaît des centaines de fois : inutile de le traiter
# des centaines de fois. On compte chaque mot UNE fois, avec sa
# fréquence, et on fait les fusions sur cette petite liste.
# ---------------------------------------------------------------
freq_mots = Counter(morceaux)

caracteres = sorted(set(texte))
vocab = {i: c for i, c in enumerate(caracteres)}   # id -> texte
car_vers_id = {c: i for i, c in vocab.items()}

# chaque mot devient un tuple d'ids : "jean" -> (14, 9, 5, 18)
mots = {tuple(car_vers_id[c] for c in m): n for m, n in freq_mots.items()}

print(f"Texte : {len(texte):,} caractères")
print(f"Morceaux (mots, ponctuation) : {len(morceaux):,}")
print(f"Morceaux DIFFÉRENTS : {len(mots):,}  <- c'est sur eux qu'on travaille")
print(f"Vocabulaire de départ : {len(vocab)} caractères\n")


def fusionner(mot, paire, nouvel_id):
    resultat = []
    i = 0
    while i < len(mot):
        if i < len(mot) - 1 and (mot[i], mot[i + 1]) == paire:
            resultat.append(nouvel_id)
            i += 2
        else:
            resultat.append(mot[i])
            i += 1
    return tuple(resultat)


def bilan(mots, vocab):
    """Mesure ce qu'on gagne (séquences plus courtes)
    et ce qu'on perd (tokens rares)."""
    usage = Counter()
    for mot, n in mots.items():
        for t in mot:
            usage[t] += n
    total = sum(usage.values())
    rares = sum(1 for t in vocab if usage[t] < 10)
    return total, rares


# ---------------------------------------------------------------
# 3. ENTRAÎNEMENT, avec un bilan à plusieurs tailles de vocabulaire
# ---------------------------------------------------------------
NB_FUSIONS = 3000
POINTS_DE_BILAN = {100, 500, 1000, 2000, 3000}
fusions = {}

print(f"{'fusions':>8} | {'vocab':>6} | {'tokens dans le texte':>21} | "
      f"{'car./token':>10} | {'tokens vus < 10 fois':>21}")
print("-" * 80)

t0 = time.time()
for k in range(1, NB_FUSIONS + 1):
    paires = Counter()
    for mot, n in mots.items():
        for p in zip(mot, mot[1:]):
            paires[p] += n
    if not paires:
        print("Plus rien à fusionner : tous les mots sont devenus des tokens.")
        break

    paire, freq = paires.most_common(1)[0]
    nouvel_id = len(vocab)
    vocab[nouvel_id] = vocab[paire[0]] + vocab[paire[1]]
    fusions[paire] = nouvel_id
    mots = {fusionner(mot, paire, nouvel_id): n for mot, n in mots.items()}

    if k in POINTS_DE_BILAN:
        total, rares = bilan(mots, vocab)
        print(f"{k:>8} | {len(vocab):>6} | {total:>21,} | "
              f"{len(texte) / total:>10.2f} | "
              f"{rares:>6} ({rares / len(vocab):.0%} du vocab)")

print(f"\nEntraînement terminé en {time.time() - t0:.0f} s")

plus_longs = sorted(vocab.values(), key=len, reverse=True)[:15]
print("\nTokens les plus longs :", plus_longs)

# ---------------------------------------------------------------
# 4. ENCODER / DÉCODER
# ---------------------------------------------------------------
def encoder(phrase):
    ids = []
    for m in motif.findall(phrase.lower()):
        mot = tuple(car_vers_id[c] for c in m if c in car_vers_id)
        for paire, nouvel_id in fusions.items():
            if len(mot) < 2:
                break
            mot = fusionner(mot, paire, nouvel_id)
        ids.extend(mot)
    return ids


def decoder(ids):
    return "".join(vocab[i] for i in ids)


phrase = "Jean Valjean regardait l'évêque avec une étrange tranquillité."
codes = encoder(phrase)
attendu = "".join(c for c in phrase.lower() if c in car_vers_id)
print(f"\nPhrase : {phrase}")
print(f"Découpage : {'|'.join(vocab[i] for i in codes)}")
print(f"{len(phrase)} caractères -> {len(codes)} tokens (contre 25 avec bpe.py)")
print(f"Décodage correct : {decoder(codes) == attendu}")
