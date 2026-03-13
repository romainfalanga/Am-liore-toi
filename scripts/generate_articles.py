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


def call_mammouth(messages, max_tokens=4000, temperature=0.85):
    """Appelle l'API Mammouth avec gemini-3-flash-preview."""
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
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=180)
    if resp.status_code != 200:
        print(f"  ERREUR API ({resp.status_code}): {resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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

def generate_article(combo):
    """Genere un article profond ou l'histoire sert intimement le propos."""

    cat = combo["categorie"]
    cat_labels = {"communication": "Communication", "productivite": "Productivite", "memorisation": "Memorisation"}
    cat_label = cat_labels[cat]

    system_prompt = f"""Tu es un auteur expert en developpement personnel, specialise en {cat_label.lower()}.
Tu ecris des articles de blog en francais d'une qualite exceptionnelle.
Ton style combine la profondeur d'un psychologue, la clarte d'un pedagogue et la chaleur d'un ami bienveillant.
Tu utilises le tutoiement pour creer de la proximite.
Chaque article doit etre un voyage transformateur pour le lecteur."""

    user_prompt = f"""Ecris un article de blog EXCEPTIONNEL pour la categorie {cat_label}.

PROFIL DU PERSONNAGE :
- Prenom : {combo['prenom']}
- Age : {combo['age_exact']} ans ({combo['tranche_age']})
- Metier/role : {combo.get('metier_ou_role', 'non precise')}
- Genre : {combo['genre']}

PARAMETRES PRECIS :
- Aspect traite : {combo['aspect']}
- Contexte de vie : {combo['contexte']}
- Problematique profonde : {combo['problematique']}

STRUCTURE OBLIGATOIRE — chaque partie doit etre INTIMEMENT LIEE aux autres :

## L'histoire de {combo['prenom']} (350-450 mots)
Raconte une histoire VIVANTE et IMMERSIVE de {combo['prenom']}, {combo['age_exact']} ans, {combo.get('metier_ou_role', '')}.
- Plante le decor avec des details sensoriels (ce qu'il/elle voit, entend, ressent physiquement)
- Montre la problematique "{combo['problematique']}" en ACTION dans le contexte "{combo['contexte']}"
- Inclus des dialogues realistes et naturels
- Montre les CONSEQUENCES concretes de cette problematique sur sa vie
- Termine par un moment declencheur : une prise de conscience, un echec cuisant, ou un feedback inattendu
- L'histoire doit etre suffisamment detaillee pour que le lecteur s'identifie profondement

## Le diagnostic : comprendre {combo['problematique']} (200-250 mots)
- Donne un nom precis a la problematique et explique-la avec des termes clairs
- Explique POURQUOI cette problematique est si frequente chez les personnes de {combo['age_exact']} ans dans ce type de situation
- Donne 3-4 signes concrets et specifiques pour reconnaitre ce probleme chez soi (en lien direct avec l'histoire de {combo['prenom']})
- Explique les consequences a moyen et long terme si on ne traite pas ce probleme
- Reference si possible un concept psychologique ou une etude qui eclaire cette problematique

## 3 solutions pour transformer ta {cat_label.lower()} (500-600 mots au total)
Chaque solution doit :
- Avoir un titre percutant et memorable
- Expliquer le POURQUOI ca fonctionne (mecanisme psychologique ou cognitif en 2-3 phrases)
- Donner un EXERCICE PRATIQUE ultra-concret faisable aujourd'hui meme
- Montrer EXACTEMENT comment {combo['prenom']} aurait pu appliquer cette solution dans sa situation specifique (retour a l'histoire)
- Inclure un exemple de resultat attendu si on applique cette solution sur 2-4 semaines

La solution 1 doit etre applicable IMMEDIATEMENT (quick win).
La solution 2 doit etre une HABITUDE a construire sur 2 semaines.
La solution 3 doit etre un CHANGEMENT DE MENTALITE profond.

## Ce que {combo['prenom']} a fait ensuite (100-150 mots)
Epilogue : raconte brievement comment {combo['prenom']} a applique UNE des solutions et le resultat concret obtenu.
Termine par une phrase inspirante qui donne envie d'agir.

REGLES D'ECRITURE :
- N'ecris PAS de titre H1 (il est ajoute automatiquement via le front matter)
- Utilise ## pour les sections principales et ### pour les sous-titres des solutions
- Le contenu total doit faire entre 1200 et 1800 mots
- Chaque phrase doit apporter de la valeur, ZERO remplissage
- Les solutions doivent etre specifiques a la tranche d'age {combo['tranche_age']} et au contexte "{combo['contexte']}"
- Utilise des metaphores et analogies pour rendre les concepts memorables
- Inclus des chiffres ou references quand c'est pertinent"""

    print(f"  Generation de l'article ({combo['prenom']}, {combo['age_exact']} ans, {cat})...")
    content = call_mammouth(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=4000,
        temperature=0.85
    )

    return content


def generate_seo(combo, article_content):
    """Genere des metadonnees SEO parfaitement optimisees."""

    prompt = f"""Genere des metadonnees SEO optimisees pour cet article de blog en francais.

Categorie: {combo['categorie']}
Aspect: {combo['aspect']}
Contexte: {combo['contexte']}
Personnage: {combo['prenom']}, {combo['age_exact']} ans

Debut de l'article:
{article_content[:600]}

Reponds UNIQUEMENT en JSON valide :
{{
  "title": "Titre SEO accrocheur de 55-65 caracteres qui donne envie de cliquer",
  "description": "Meta description de 150-160 caracteres avec appel a l'action",
  "slug": "slug-court-en-minuscules-sans-accents-avec-tirets",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

Le titre doit :
- Contenir le mot-cle principal lie a "{combo['aspect']}"
- Evoquer un benefice clair pour le lecteur
- Etre naturel et engageant (pas de clickbait grossier)

Les tags doivent etre en francais, sans accents, pertinents pour le SEO."""

    response = call_mammouth(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )

    try:
        seo = extract_json_from_response(response)
        if isinstance(seo, list):
            seo = seo[0]
        return seo
    except (ValueError, IndexError):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = re.sub(r'[^a-z0-9-]', '', combo['aspect'][:40].lower().replace(' ', '-').replace("'", ''))
        return {
            "title": f"{combo['aspect'].title()} : transformez votre {combo['categorie']}",
            "description": f"Decouvrez comment surmonter {combo['problematique'][:60]}. 3 solutions concretes.",
            "slug": f"{combo['categorie']}-{slug}-{today}",
            "tags": [combo['categorie'], "developpement-personnel", "amelioration"]
        }


# ============================================================
# PHASE 4 : CREATION DU FICHIER ET MISE A JOUR DU REGISTRE
# ============================================================

def create_hugo_article(combo, seo, article_content, today):
    """Cree le fichier Markdown Hugo."""
    cat = combo["categorie"]

    slug = re.sub(r'[^a-z0-9-]', '', seo.get("slug", "").lower().replace(" ", "-"))
    if not slug or len(slug) < 5:
        slug = f"{cat}-{combo['prenom'].lower()}-{today}"

    filename = f"{today}-{slug}.md"
    filepath = CONTENT_DIR / cat / filename

    tags_str = ", ".join(f'"{t}"' for t in seo.get("tags", []))
    title_safe = seo.get("title", "").replace('"', "'")
    desc_safe = seo.get("description", "").replace('"', "'")

    front_matter = f"""---
title: "{title_safe}"
date: {today}T08:00:00+01:00
description: "{desc_safe}"
categories: ["{cat.title()}"]
tags: [{tags_str}]
slug: "{slug}"
draft: false
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
    for combo in combinations:
        print(f"  [{combo['categorie'].upper()}] {combo['prenom']} ({combo['age_exact']} ans) — {combo['aspect'][:50]}...")

    # --- PHASE 3 : Generation ---
    print("\n[PHASE 3] Generation des articles...")
    for combo in combinations:
        cat = combo["categorie"]
        print(f"\n--- {cat.upper()} ---")

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

    # --- PHASE 4 : Sauvegarde ---
    print("\n[PHASE 4] Mise a jour du registre...")
    save_json(REGISTRY_PATH, registry)

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
    print("  Generation terminee avec succes !")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
