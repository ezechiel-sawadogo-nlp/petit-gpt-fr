import time
from collections import Counter

# ---------------------------------------------------------------
# 0. LIRE ET NETTOYER LE TEXTE
# On retire la licence Project Gutenberg (en anglais) au début
# et à la fin : c'est elle qui faisait apparaître "other",
# "projection"... dans tes textes générés.
# ---------------------------------------------------------------
with open("texte.txt", encoding="utf-8") as f:
    texte = f.read()

debut = texte.find("*** START OF")
fin = texte.find("*** END OF")
if debut != -1 and fin != -1:
    texte = texte[texte.find("\n", debut) + 1 : fin]

texte = texte.lower()
texte = texte[:200_000]   # un morceau suffit pour apprendre, et c'est plus rapide

# ---------------------------------------------------------------
# 1. VOCABULAIRE DE DÉPART : un token = un caractère
# Chaque caractère reçoit un numéro (son "id").
# Le texte devient une liste de nombres : c'est ça qu'un modèle lit.
# ---------------------------------------------------------------
caracteres = sorted(set(texte))
vocab = {i: c for i, c in enumerate(caracteres)}   # id -> texte
car_vers_id = {c: i for i, c in vocab.items()}     # texte -> id
ids = [car_vers_id[c] for c in texte]

print(f"Vocabulaire de départ : {len(vocab)} tokens (les caractères)")
print(f"Texte = {len(ids):,} tokens\n")

# ---------------------------------------------------------------
# 2. LES DEUX OPÉRATIONS DU BPE
# ---------------------------------------------------------------
def compter_paires(ids):
    """Compte combien de fois chaque paire de tokens voisins apparaît."""
    return Counter(zip(ids, ids[1:]))

def fusionner(ids, paire, nouvel_id):
    """Remplace chaque occurrence de `paire` par un seul token `nouvel_id`."""
    resultat = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == paire:
            resultat.append(nouvel_id)
            i += 2
        else:
            resultat.append(ids[i])
            i += 1
    return resultat

# ---------------------------------------------------------------
# 3. ENTRAÎNEMENT : on répète "trouver la paire la plus fréquente,
# la fusionner en un nouveau token" NB_FUSIONS fois.
# ---------------------------------------------------------------
NB_FUSIONS = 300
fusions = {}   # (id_a, id_b) -> nouvel_id, dans l'ordre où on les apprend

t0 = time.time()
for k in range(NB_FUSIONS):
    paires = compter_paires(ids)
    paire, freq = paires.most_common(1)[0]
    nouvel_id = len(vocab)
    ids = fusionner(ids, paire, nouvel_id)
    fusions[paire] = nouvel_id
    vocab[nouvel_id] = vocab[paire[0]] + vocab[paire[1]]

    if k < 15 or (k + 1) % 50 == 0:
        a, b, nouveau = vocab[paire[0]], vocab[paire[1]], vocab[nouvel_id]
        print(f"fusion {k + 1:>3} : {a!r:>8} + {b!r:<8} -> {nouveau!r:<12} ({freq} fois)")

print(f"\nEntraînement terminé en {time.time() - t0:.0f} s")
print(f"Vocabulaire final : {len(vocab)} tokens")
print(f"Texte = {len(ids):,} tokens (au lieu de {len(texte):,} caractères)")
print(f"Compression : {len(texte) / len(ids):.2f} caractères par token en moyenne")

# Les tokens les plus longs que le BPE a construits
plus_longs = sorted(vocab.values(), key=len, reverse=True)[:15]
print("\nTokens les plus longs :", [t for t in plus_longs])

# ---------------------------------------------------------------
# 4. ENCODER ET DÉCODER UNE NOUVELLE PHRASE
# Encoder : texte -> liste d'ids (on rejoue les fusions dans l'ordre).
# Décoder : liste d'ids -> texte (on recolle les morceaux).
# ---------------------------------------------------------------
def encoder(phrase):
    ids = [car_vers_id[c] for c in phrase.lower() if c in car_vers_id]
    for paire, nouvel_id in fusions.items():
        ids = fusionner(ids, paire, nouvel_id)
    return ids

def decoder(ids):
    return "".join(vocab[i] for i in ids)

phrase = "Jean Valjean regardait l'évêque avec une étrange tranquillité."
codes = encoder(phrase)
print(f"\nPhrase : {phrase}")
print(f"Ids    : {codes}")
print(f"Découpage : {'|'.join(vocab[i] for i in codes)}")
print(f"{len(phrase)} caractères -> {len(codes)} tokens")
print(f"Décodage correct : {decoder(codes) == phrase.lower()}")
