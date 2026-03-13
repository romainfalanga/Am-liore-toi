#!/usr/bin/env python3
"""
Ameliore-toi — Generateur d'articles autonome a matrice infinie.

Pipeline en 4 phases :
  1. ANALYSE  : Lire tous les articles existants, extraire ce qui a deja ete couvert
  2. PROPOSITION : L'IA propose 3 combinaisons 100% nouvelles et coherentes
  3. GENERATION : Articles profonds avec histoire intimement liee aux solutions
  4. MISE A JOUR : Le registre est complete pour ne jamais reproduire un article
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- Configuration ---
API_URL = "https://api.mammouth.ai/v1/chat/completions"
API_KEY = os.environ.get("MAMMOUTH_API_KEY", "")
MODEL = "gemini-3-flash-preview"

BASE_DIR = Path(__file__).resolve().parent.parent
MATRIX_PATH = BASE_DIR / "data" / "matrix.json"
REGISTRY_PATH = BASE_DIR / "data" / "article_registry.json"
CONTENT_DIR = BASE_DIR / "content"

CATEGORIES = ["communication", "productivite", "memorisation"]


# ============================================================
# UTILITAIRES
# ============================================================

def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def call_mammouth(messages, max_tokens=4000, temperature=0.85, retries=3):
    """Appelle l'API Mammouth avec retry automatique en cas d'echec."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=180)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            print(f"  ERREUR API tentative {attempt}/{retries} ({resp.status_code}): {resp.text[:300]}")
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"  ERREUR reseau tentative {attempt}/{retries}: {e}")
        if attempt < retries:
            wait = 2 ** attempt
            print(f"  Nouvelle tentative dans {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Echec API apres {retries} tentatives. Derniere erreur: {last_error}")


def extract_json_from_response(text):
    """Extrait un objet ou tableau JSON d'une reponse qui peut contenir du texte autour."""
    # Essayer de trouver un bloc ```json ... ```
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Essayer de trouver un tableau JSON
    match = re.search(r'(\[[\s\S]*\])', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Essayer de trouver un objet JSON
    match = re.search(r'(\{[\s\S]*\})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Impossible d'extraire du JSON de la reponse:\n{text[:300]}")


# ============================================================
# PHASE 1 : ANALYSE DES ARTICLES EXISTANTS
# ============================================================

def load_registry():
    """Charge le registre de tous les articles generes."""
    registry = load_json(REGISTRY_PATH)
    if registry is None:
        registry = {
            "articles": [],
            "prenoms_utilises": [],
            "combinaisons_utilisees": []
        }
    return registry


def scan_existing_articles():
    """Parcourt tous les fichiers markdown pour extraire les metadonnees."""
    articles_summary = {cat: [] for cat in CATEGORIES}

    for cat in CATEGORIES:
        cat_dir = CONTENT_DIR / cat
        if not cat_dir.exists():
            continue
        for md_file in sorted(cat_dir.glob("*.md")):
            if md_file.name == "_index.md":
                continue
            content = md_file.read_text(encoding="utf-8")
            # Extraire le titre du front matter
            title_match = re.search(r'^title:\s*"(.+)"', content, re.MULTILINE)
            title = title_match.group(1) if title_match else md_file.stem
            # Extraire les tags
            tags_match = re.search(r'^tags:\s*\[(.+)\]', content, re.MULTILINE)
            tags = tags_match.group(1) if tags_match else ""
            # Extraire la description
            desc_match = re.search(r'^description:\s*"(.+)"', content, re.MULTILINE)
            desc = desc_match.group(1) if desc_match else ""

            articles_summary[cat].append({
                "file": md_file.name,
                "title": title,
                "tags": tags,
                "description": desc
            })

    return articles_summary


# ============================================================
# PHASE 2 : PROPOSITION DE COMBINAISONS UNIQUES PAR L'IA
# ============================================================

def propose_combinations(matrix, registry, existing_articles):
    """Demande a l'IA de proposer 3 combinaisons uniques et coherentes."""

    prenoms_deja_utilises = registry.get("prenoms_utilises", [])
    combinaisons_deja_faites = registry.get("combinaisons_utilisees", [])

    # Construire le resume des articles existants par categorie
    existing_summary = ""
    for cat in CATEGORIES:
        articles = existing_articles.get(cat, [])
        if articles:
            existing_summary += f"\n--- {cat.upper()} ({len(articles)} articles existants) ---\n"
            for a in articles[-20:]:  # Les 20 derniers pour limiter la taille
                existing_summary += f"  - {a['title']}\n"
        else:
            existing_summary += f"\n--- {cat.upper()} (aucun article existant) ---\n"

    # Construire la liste des combinaisons deja faites
    combos_text = ""
    if combinaisons_deja_faites:
        combos_text = "\n\nCOMBINAISONS DEJA REALISEES (NE PAS REPRODUIRE) :\n"
        for c in combinaisons_deja_faites[-60:]:  # Les 60 dernieres
            combos_text += f"  - [{c['categorie']}] {c['aspect']} | {c['contexte']} | {c['problematique']} | {c['prenom']} ({c['age']})\n"

    # Prenoms deja utilises
    prenoms_text = ", ".join(prenoms_deja_utilises[-100:]) if prenoms_deja_utilises else "aucun"

    prompt = f"""Tu es le cerveau strategique du blog "Ameliore-toi". Tu dois proposer exactement 3 articles — un par categorie (communication, productivite, memorisation) — qui n'ont JAMAIS ete faits.

ARTICLES EXISTANTS SUR LE BLOG :
{existing_summary}
{combos_text}

PRENOMS DEJA UTILISES (ne pas reutiliser) : {prenoms_text}

MATRICE DISPONIBLE — voici les dimensions pour chaque categorie. Tu DOIS piocher dans ces listes :

TRANCHES D'AGE DISPONIBLES :
{json.dumps(matrix['tranches_age'], ensure_ascii=False, indent=2)}

COMMUNICATION — aspects :
{json.dumps(matrix['communication']['aspects'], ensure_ascii=False)}

COMMUNICATION — contextes :
{json.dumps(matrix['communication']['contextes_specifiques'], ensure_ascii=False)}

COMMUNICATION — problematiques :
{json.dumps(matrix['communication']['problematiques_profondes'], ensure_ascii=False)}

PRODUCTIVITE — aspects :
{json.dumps(matrix['productivite']['aspects'], ensure_ascii=False)}

PRODUCTIVITE — contextes :
{json.dumps(matrix['productivite']['contextes_specifiques'], ensure_ascii=False)}

PRODUCTIVITE — problematiques :
{json.dumps(matrix['productivite']['problematiques_profondes'], ensure_ascii=False)}

MEMORISATION — aspects :
{json.dumps(matrix['memorisation']['aspects'], ensure_ascii=False)}

MEMORISATION — contextes :
{json.dumps(matrix['memorisation']['contextes_specifiques'], ensure_ascii=False)}

MEMORISATION — problematiques :
{json.dumps(matrix['memorisation']['problematiques_profondes'], ensure_ascii=False)}

PRENOMS DISPONIBLES (masculins) : {json.dumps(matrix['prenoms']['masculins'], ensure_ascii=False)}
PRENOMS DISPONIBLES (feminins) : {json.dumps(matrix['prenoms']['feminins'], ensure_ascii=False)}

REGLES IMPERATIVES :
1. Choisis une combinaison aspect + contexte + problematique + tranche d'age + prenom QUI N'A JAMAIS ETE FAITE
2. Le prenom DOIT etre different de tous les prenoms deja utilises
3. La combinaison doit etre COHERENTE : l'aspect doit avoir du sens dans le contexte, la problematique doit etre pertinente pour cette situation, et l'age doit correspondre logiquement au contexte
4. Alterne les genres (masculin/feminin) entre les 3 articles
5. Varie les tranches d'age entre les 3 articles
6. Maximise la DIVERSITE par rapport aux articles existants

Reponds UNIQUEMENT avec un tableau JSON de 3 objets, sans aucun texte autour :
[
  {{
    "categorie": "communication",
    "aspect": "...",
    "contexte": "...",
    "problematique": "...",
    "prenom": "...",
    "genre": "masculin ou feminin",
    "tranche_age": "...",
    "age_exact": 32,
    "metier_ou_role": "..."
  }},
  {{
    "categorie": "productivite",
    ...
  }},
  {{
    "categorie": "memorisation",
    ...
  }}
]"""

    print("  Appel IA pour proposer 3 combinaisons uniques...")
    response = call_mammouth(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.9
    )

    combinations = extract_json_from_response(response)

    if not isinstance(combinations, list) or len(combinations) != 3:
        raise ValueError(f"L'IA n'a pas retourne 3 combinaisons. Reponse: {response[:300]}")

    return combinations


# ============================================================
# PHASE 3 : GENERATION D'ARTICLES PROFONDS
# ============================================================

def clean_ai_artifacts(text):
    """Nettoie les artefacts typiques de l'ecriture IA dans le texte genere."""
    # Supprimer les tirets cadratins utilises comme ponctuation
    # Remplacer " — " par ", " ou " : " selon le contexte
    text = re.sub(r'\s*—\s*', ', ', text)
    # Remplacer les tirets demi-cadratins utilises comme ponctuation (pas dans les listes)
    text = re.sub(r'(?<!\n)(?<!^)(?<![0-9])\s+–\s+', ', ', text)
    # Supprimer les doubles espaces residuels
    text = re.sub(r'  +', ' ', text)
    # Nettoyer ", ," ou ": ," residuels
    text = re.sub(r',\s*,', ',', text)
    # Supprimer les emojis courants dans le texte IA
    text = re.sub(r'[🎯🔑💡🧠✨🚀💪🎓📌⚡🏆🌟❤️👉✅❌⭐🔥💥🎉🎊]', '', text)
    return text


def generate_article(combo):
    """Genere un article profond, riche et a forte valeur ajoutee."""

    cat = combo["categorie"]
    cat_labels = {"communication": "Communication", "productivite": "Productivite", "memorisation": "Memorisation"}
    cat_label = cat_labels[cat]

    prenom = combo['prenom']
    age = combo['age_exact']
    genre = combo['genre']
    pronom_il = "il" if genre == "masculin" else "elle"
    pronom_son = "son" if genre == "masculin" else "sa" if genre == "feminin" else "son"
    metier = combo.get('metier_ou_role', 'non precise')

    system_prompt = f"""Tu es un auteur expert en developpement personnel, specialise en {cat_label.lower()}.
Tu ecris des articles de blog en francais d'une qualite exceptionnelle.
Ton style combine la rigueur d'un psychologue clinicien, la clarte d'un pedagogue experimente et la chaleur d'un ami bienveillant.
Tu utilises le tutoiement pour creer une proximite naturelle avec le lecteur.

REGLES DE STYLE ABSOLUES :
- Tu n'utilises JAMAIS le tiret cadratin (—) ni le tiret demi-cadratin (–) comme ponctuation.
- Tu reformules toujours avec des virgules, des points, des deux-points ou des points-virgules.
- Tu evites les formulations generiques et les phrases creuses. Chaque phrase doit enseigner quelque chose ou faire ressentir une emotion.
- Tu ne commences JAMAIS une phrase par "En fait", "En effet", "Il est important de noter", "Force est de constater", "Il convient de".
- Tu evites les listes a puces quand une prose fluide est plus naturelle.
- Tu ecris comme un humain qui a vecu, pas comme un assistant qui compile des informations.
- Tu varies la longueur de tes phrases : certaines courtes et percutantes, d'autres longues et immersives.
- Tu utilises des transitions naturelles entre les paragraphes, jamais de formules mecaniques."""

    user_prompt = f"""Ecris un article de blog EXCEPTIONNEL et COMPLET pour la categorie {cat_label}.
Cet article doit etre une reference sur le sujet, le genre de contenu qu'on met en favori pour y revenir.

PROFIL DU PERSONNAGE :
- Prenom : {prenom}
- Age : {age} ans ({combo['tranche_age']})
- Metier/role : {metier}
- Genre : {genre}

PARAMETRES PRECIS :
- Aspect traite : {combo['aspect']}
- Contexte de vie : {combo['contexte']}
- Problematique profonde : {combo['problematique']}

STRUCTURE OBLIGATOIRE (respecte scrupuleusement ces titres de sections) :

## L'histoire de {prenom}

Raconte une histoire VIVANTE, IMMERSIVE et DETAILLEE de {prenom}, {age} ans, {metier}. Cette section doit faire entre 500 et 700 mots.
- Plante le decor avec des details sensoriels precis : la lumiere dans la piece, les sons ambiants, ce que {prenom} ressent physiquement (gorge nouee, mains moites, tension dans les epaules).
- Montre la problematique "{combo['problematique']}" en ACTION a travers une scene concrete dans le contexte "{combo['contexte']}".
- Integre des dialogues realistes et naturels, avec les hesitations et les non-dits qui rendent une conversation vraie.
- Developpe les pensees interieures de {prenom} : ce qu'{pronom_il} se dit, les doutes qui tournent en boucle, les strategies d'evitement.
- Montre les CONSEQUENCES en cascade : comment ce probleme affecte {pronom_son} travail, ses relations, {pronom_son} estime de soi, {pronom_son} sommeil.
- Integre un second personnage significatif (collegue, proche, mentor) qui joue un role dans la prise de conscience.
- Termine par un moment declencheur puissant : un echec cuisant, un feedback brutal, ou une revelation inattendue qui force {prenom} a ouvrir les yeux.
- Le lecteur doit se dire "c'est exactement ce que je vis" en lisant cette histoire.

## Pourquoi ce probleme te concerne aussi

Cette section fait entre 400 et 500 mots. Elle est le pont entre l'histoire et les solutions.
- Interpelle directement le lecteur : fais le lien entre l'experience de {prenom} et ce que le lecteur vit probablement au quotidien.
- Donne un nom precis au phenomene psychologique en jeu. Cite le concept, son auteur si possible, et explique le en termes accessibles.
- Explique les mecanismes inconscients qui entretiennent cette problematique : pourquoi on tombe dans ce piege malgre notre bonne volonte.
- Presente 4 signes concrets et specifiques pour reconnaitre ce schema chez soi. Chaque signe doit etre decrit en 2 a 3 phrases avec un exemple du quotidien.
- Decris les consequences a moyen terme (3 a 6 mois) et a long terme (1 a 3 ans) si rien ne change. Sois precis et realiste, sans dramatiser.
- Appuie ton propos sur au moins une etude, une statistique ou une reference scientifique credible.

## La methode pour transformer ta {cat_label.lower()}

Introduis cette section avec un court paragraphe (3 a 4 phrases) qui explique la philosophie derriere les solutions : pourquoi elles fonctionnent et comment elles se completent les unes les autres.

### Strategie 1 : [Titre concret et actionnable]

Cette premiere strategie est un quick win applicable des aujourd'hui. Elle doit faire 250 a 350 mots.
- Explique le PRINCIPE en 2 a 3 phrases claires.
- Decris le mecanisme psychologique ou cognitif qui rend cette technique efficace. Pourquoi ca fonctionne au niveau du cerveau ?
- Donne un PROTOCOLE PRECIS, etape par etape, que le lecteur peut suivre sans interpretation.
- Reviens a l'histoire de {prenom} : montre exactement comment {pronom_il} aurait pu appliquer cette strategie dans la scene decrite plus haut. Ecris la scene alternative.
- Decris le resultat attendu apres 7 jours d'application reguliere.

### Strategie 2 : [Titre concret et actionnable]

Cette deuxieme strategie est une habitude a construire sur 2 a 4 semaines. Elle doit faire 250 a 350 mots.
- Meme structure que la strategie 1.
- Explique pourquoi cette habitude necessite du temps et comment le cerveau s'y adapte progressivement.
- Donne un planning precis : que faire la semaine 1, la semaine 2, etc.
- Montre comment {prenom} aurait integre cette habitude dans {pronom_son} routine quotidienne.
- Decris les resultats attendus apres 3 a 4 semaines.

### Strategie 3 : [Titre concret et actionnable]

Cette troisieme strategie est un changement de perspective profond. Elle doit faire 250 a 350 mots.
- Explique la croyance limitante que cette strategie vient deconstruire.
- Propose une nouvelle facon de voir la situation, avec une analogie ou une metaphore memorable.
- Donne un exercice de reflexion ou d'ecriture que le lecteur peut faire en 15 a 20 minutes.
- Montre le basculement mental que {prenom} a du operer pour sortir de {pronom_son} schema.
- Decris la transformation profonde que ce changement de mentalite produit sur 2 a 3 mois.

## Ce que {prenom} a decide de changer

Epilogue de 200 a 300 mots. Raconte comment {prenom} a mis en pratique les strategies.
- Montre {prenom} 3 mois plus tard dans une situation similaire a celle du debut, mais avec un comportement radicalement different.
- Decris les resultats concrets et mesurables qu'{pronom_il} a obtenus.
- Termine par un paragraphe qui s'adresse directement au lecteur, l'encourage a passer a l'action sans pression, et lui rappelle qu'un premier petit pas suffit.
- La derniere phrase doit etre inspirante et memorable, le genre de phrase qu'on a envie de noter quelque part.

REGLES D'ECRITURE IMPERATIVES :
- N'ecris PAS de titre H1 (il est genere automatiquement via le front matter).
- Utilise ## pour les 4 sections principales et ### pour les 3 strategies uniquement.
- Le contenu total doit faire entre 2500 et 3500 mots. C'est un article de fond, pas un billet rapide.
- ZERO phrase de remplissage. Si une phrase n'enseigne pas ou ne fait pas ressentir, supprime la.
- N'utilise JAMAIS le tiret cadratin (—) ni le tiret demi-cadratin (–) comme ponctuation. Utilise des virgules, des points, des deux-points ou des points-virgules.
- N'utilise AUCUN emoji dans le texte.
- Ne commence jamais un paragraphe par "Il est important", "Force est de constater", "En effet", "En fait", "Notons que".
- Les solutions doivent etre ultra-specifiques a la tranche d'age {combo['tranche_age']} et au contexte "{combo['contexte']}".
- Utilise des metaphores et analogies originales pour rendre les concepts memorables.
- Integre des chiffres, des references scientifiques ou des etudes quand c'est pertinent.
- Les dialogues doivent sonner vrais : hesitations, phrases inachevees, langage du quotidien.
- Chaque section doit se terminer par une transition naturelle vers la suivante."""

    print(f"  Generation de l'article ({prenom}, {age} ans, {cat})...")
    content = call_mammouth(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=8000,
        temperature=0.82
    )

    # Post-traitement : nettoyer les artefacts IA
    content = clean_ai_artifacts(content)

    return content


def generate_seo(combo, article_content):
    """Genere des metadonnees SEO parfaitement optimisees."""

    # Extraire les titres H2/H3 pour generer la FAQ
    headings = re.findall(r'^#{2,3}\s+(.+)$', article_content, re.MULTILINE)
    headings_text = "\n".join(f"  - {h}" for h in headings) if headings else "Aucun titre extrait"

    prompt = f"""Tu es un expert SEO francophone. Genere des metadonnees SEO optimisees pour cet article de blog.

CATEGORIE : {combo['categorie']}
ASPECT TRAITE : {combo['aspect']}
CONTEXTE : {combo['contexte']}
PROBLEMATIQUE : {combo['problematique']}
PERSONNAGE : {combo['prenom']}, {combo['age_exact']} ans, {combo.get('metier_ou_role', '')}

SECTIONS DE L'ARTICLE :
{headings_text}

DEBUT DE L'ARTICLE :
{article_content[:1000]}

Reponds UNIQUEMENT en JSON valide, sans texte autour :
{{
  "title": "Titre SEO de 50 a 65 caracteres, mot-cle principal en debut de titre",
  "description": "Meta description de 145 a 160 caracteres. Doit contenir le mot-cle principal, un benefice clair et un appel a l'action implicite. Ne pas utiliser de tiret cadratin.",
  "slug": "slug-en-minuscules-sans-accents-4-a-7-mots-maximum",
  "tags": ["mot-cle-principal", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
  "keywords": ["mot cle principal", "mot cle secondaire 1", "mot cle secondaire 2", "mot cle longue traine 1", "mot cle longue traine 2"],
  "faq": [
    {{"question": "Question naturelle que poserait un internaute sur Google en lien avec le sujet", "answer": "Reponse concise de 2 a 3 phrases qui apporte une vraie valeur."}},
    {{"question": "Deuxieme question naturelle differente de la premiere", "answer": "Reponse concise de 2 a 3 phrases."}},
    {{"question": "Troisieme question naturelle", "answer": "Reponse concise de 2 a 3 phrases."}}
  ]
}}

REGLES POUR LE TITRE :
- Le mot-cle principal lie a "{combo['aspect']}" doit apparaitre dans les 40 premiers caracteres
- Le titre doit evoquer un benefice concret ou un resultat
- Pas de clickbait, pas de "vous n'allez pas croire..."
- Ne pas utiliser de tiret cadratin (—) ni de tiret demi-cadratin (–), utiliser " : " si besoin

REGLES POUR LES TAGS :
- 7 tags maximum, en francais, sans accents
- Le premier tag est le mot-cle principal
- Inclure des variantes longue traine pertinentes pour le SEO
- Tags specifiques au contexte ({combo['contexte']}) et a la problematique

REGLES POUR LES KEYWORDS :
- 5 mots-cles au total : 1 principal + 2 secondaires + 2 longue traine
- Les mots-cles longue traine doivent correspondre a des requetes reelles que taperait un internaute

REGLES POUR LA FAQ :
- 3 questions que quelqu'un poserait naturellement sur Google a propos de ce sujet
- Les reponses doivent etre concises (2 a 3 phrases), precises et utiles
- Ne pas repeter le contenu de l'article mot pour mot, reformuler
- Ne pas utiliser de tiret cadratin (—) ni de tiret demi-cadratin (–)"""

    response = call_mammouth(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.3
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    aspect_slug = re.sub(r'[^a-z0-9-]', '', combo['aspect'][:40].lower().replace(' ', '-').replace(chr(39), ''))
    fallback = {
        "title": f"{combo['aspect'].title()} : transformez votre {combo['categorie']}",
        "description": f"Decouvrez comment surmonter {combo['problematique'][:60]}. Methode complete avec 3 strategies concretes et un plan d'action.",
        "slug": f"{combo['categorie']}-{aspect_slug}-{today}",
        "tags": [combo['categorie'], "developpement-personnel", "amelioration", combo['aspect'].lower().replace(' ', '-')[:30]],
        "keywords": [combo['aspect'].lower(), combo['categorie'], "developpement personnel"],
        "faq": []
    }

    try:
        seo = extract_json_from_response(response)
        if isinstance(seo, list):
            seo = seo[0]
        if not isinstance(seo, dict):
            print(f"  SEO: reponse non-dict ({type(seo).__name__}), utilisation du fallback")
            return fallback
        # Valider les cles essentielles
        for key in ("title", "description", "slug"):
            if key not in seo or not isinstance(seo[key], str):
                seo[key] = fallback[key]
        if "tags" not in seo or not isinstance(seo["tags"], list):
            seo["tags"] = fallback["tags"]
        if "keywords" not in seo or not isinstance(seo["keywords"], list):
            seo["keywords"] = fallback["keywords"]
        if "faq" not in seo or not isinstance(seo["faq"], list):
            seo["faq"] = []
        # Nettoyer les tirets cadratins dans le titre et la description
        seo["title"] = seo["title"].replace("—", " : ").replace("–", " : ")
        seo["description"] = seo["description"].replace("—", ", ").replace("–", ", ")
        return seo
    except Exception as e:
        print(f"  SEO: erreur parsing ({e}), utilisation du fallback")
        return fallback


# ============================================================
# PHASE 4 : CREATION DU FICHIER ET MISE A JOUR DU REGISTRE
# ============================================================

def create_hugo_article(combo, seo, article_content, today):
    """Cree le fichier Markdown Hugo avec front matter SEO enrichi."""
    cat = combo["categorie"]

    slug = re.sub(r'[^a-z0-9-]', '', seo.get("slug", "").lower().replace(" ", "-"))
    if not slug or len(slug) < 5:
        slug = f"{cat}-{combo['prenom'].lower()}-{today}"

    filename = f"{today}-{slug}.md"
    filepath = CONTENT_DIR / cat / filename

    tags_str = ", ".join(f'"{t}"' for t in seo.get("tags", []))
    keywords_str = ", ".join(f'"{k}"' for k in seo.get("keywords", []))
    title_safe = seo.get("title", "").replace('"', "'")
    desc_safe = seo.get("description", "").replace('"', "'")

    # Calculer le temps de lecture
    word_count = len(article_content.split())
    read_time = max(1, round(word_count / 200))

    # Generer la section FAQ en front matter pour le schema JSON-LD
    faq_items = seo.get("faq", [])
    faq_yaml = ""
    if faq_items:
        faq_yaml = "faq:\n"
        for item in faq_items:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                q = item["question"].replace('"', "'")
                a = item["answer"].replace('"', "'")
                faq_yaml += f'  - question: "{q}"\n    answer: "{a}"\n'

    front_matter = f"""---
title: "{title_safe}"
date: {today}T08:00:00+01:00
lastmod: {today}T08:00:00+01:00
description: "{desc_safe}"
categories: ["{cat.title()}"]
tags: [{tags_str}]
keywords: [{keywords_str}]
slug: "{slug}"
readingTime: {read_time}
wordCount: {word_count}
author: "Ameliore-toi"
{faq_yaml}draft: false
---

"""

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(front_matter + article_content)

    print(f"  -> Article cree : {filepath.name}")
    return filepath


def update_registry(registry, combo, seo, today):
    """Met a jour le registre pour ne jamais reproduire un article."""
    registry["articles"].append({
        "date": today,
        "categorie": combo["categorie"],
        "titre": seo.get("title", ""),
        "prenom": combo["prenom"],
        "age": combo["age_exact"],
        "tranche_age": combo["tranche_age"],
        "aspect": combo["aspect"],
        "contexte": combo["contexte"],
        "problematique": combo["problematique"],
        "slug": seo.get("slug", "")
    })

    registry["prenoms_utilises"].append(combo["prenom"])

    registry["combinaisons_utilisees"].append({
        "categorie": combo["categorie"],
        "aspect": combo["aspect"],
        "contexte": combo["contexte"],
        "problematique": combo["problematique"],
        "prenom": combo["prenom"],
        "age": combo["tranche_age"]
    })


# ============================================================
# MAIN
# ============================================================

def main():
    if not API_KEY:
        raise RuntimeError("MAMMOUTH_API_KEY non definie. Ajoutez-la dans les secrets GitHub.")

    print("=" * 70)
    print(f"  AMELIORE-TOI — Generation d'articles")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 70)

    matrix = load_json(MATRIX_PATH)
    registry = load_registry()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- PHASE 1 : Analyse ---
    print("\n[PHASE 1] Analyse des articles existants...")
    existing_articles = scan_existing_articles()
    for cat in CATEGORIES:
        count = len(existing_articles.get(cat, []))
        print(f"  {cat}: {count} articles")
    print(f"  Prenoms deja utilises: {len(registry.get('prenoms_utilises', []))}")
    print(f"  Combinaisons faites: {len(registry.get('combinaisons_utilisees', []))}")

    # --- PHASE 2 : Proposition ---
    print("\n[PHASE 2] Proposition de 3 combinaisons uniques...")
    combinations = propose_combinations(matrix, registry, existing_articles)

    # Verifier que les 3 categories sont presentes, sinon reordonner
    categories_in_combos = [c["categorie"] for c in combinations]
    for expected_cat in CATEGORIES:
        if expected_cat not in categories_in_combos:
            print(f"  ATTENTION: categorie '{expected_cat}' manquante dans les combinaisons!")

    for combo in combinations:
        print(f"  [{combo['categorie'].upper()}] {combo['prenom']} ({combo['age_exact']} ans) — {combo['aspect'][:50]}...")

    # --- PHASE 3 : Generation ---
    print("\n[PHASE 3] Generation des articles...")
    success_count = 0
    failed_categories = []

    for combo in combinations:
        cat = combo["categorie"]
        print(f"\n--- {cat.upper()} ---")

        try:
            # Generer l'article
            article_content = generate_article(combo)

            # Generer le SEO
            print(f"  Generation du SEO...")
            seo = generate_seo(combo, article_content)
            print(f"  Titre: {seo.get('title', 'N/A')}")

            # Creer le fichier
            create_hugo_article(combo, seo, article_content, today)

            # Mettre a jour le registre
            update_registry(registry, combo, seo, today)

            # Sauvegarder le registre apres chaque article reussi
            save_json(REGISTRY_PATH, registry)
            print(f"  Registre mis a jour.")

            success_count += 1

        except Exception as e:
            print(f"  ERREUR pour {cat.upper()}: {e}")
            print(f"  L'article {cat} n'a pas pu etre genere. Poursuite avec les autres categories...")
            failed_categories.append(cat)
            continue

    # --- PHASE 4 : Bilan ---
    print(f"\n[PHASE 4] Bilan de la generation...")
    print(f"  Articles generes avec succes: {success_count}/3")

    if failed_categories:
        print(f"  Categories en echec: {', '.join(failed_categories)}")

    total_articles = len(registry["articles"])
    total_prenoms = len(set(registry["prenoms_utilises"]))
    print(f"  Total articles generes: {total_articles}")
    print(f"  Total prenoms uniques: {total_prenoms}")

    # Calcul du potentiel restant
    total_prenoms_dispo = len(matrix["prenoms"]["masculins"]) + len(matrix["prenoms"]["feminins"])
    prenoms_restants = total_prenoms_dispo - total_prenoms
    combos_par_cat = 35 * 30 * 20 * 6  # aspects * contextes * problematiques * ages
    print(f"  Prenoms restants: {prenoms_restants}/{total_prenoms_dispo}")
    print(f"  Combinaisons possibles par categorie: {combos_par_cat:,}")
    print(f"  Potentiel total: {combos_par_cat * 3:,} articles uniques")

    print(f"\n{'=' * 70}")
    if failed_categories:
        print(f"  Generation terminee avec {len(failed_categories)} echec(s): {', '.join(failed_categories)}")
        print(f"  Les {success_count} autres articles ont ete generes et sauvegardes.")
    else:
        print("  Generation terminee avec succes !")
    print(f"{'=' * 70}")

    # Ne pas faire echouer le workflow si au moins 1 article a ete genere
    if success_count == 0:
        raise RuntimeError("Aucun article n'a pu etre genere. Verifiez l'API et les logs.")


if __name__ == "__main__":
    main()
