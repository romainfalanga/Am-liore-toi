#!/usr/bin/env python3
"""
Script de generation automatique d'articles pour Ameliore-toi.
Genere 3 articles quotidiens (communication, productivite, memorisation)
en utilisant l'API Mammouth avec un systeme de matrice combinatoire.
"""

import json
import os
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- Configuration ---
API_URL = "https://api.mammouth.ai/v1/chat/completions"
API_KEY = os.environ.get("MAMMOUTH_API_KEY", "")
# Modele pour la generation d'articles et le SEO
MODEL_ARTICLE = "gemini-3-flash-preview"
MODEL_SEO = "gemini-3-flash-preview"

BASE_DIR = Path(__file__).resolve().parent.parent
MATRIX_PATH = BASE_DIR / "data" / "matrix.json"
STATE_PATH = BASE_DIR / "data" / "generation_state.json"

CATEGORIES = ["communication", "productivite", "memorisation"]


def load_matrix():
    with open(MATRIX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_indices": {cat: {"competence": 0, "situation": 0, "problematique": 0} for cat in CATEGORIES},
            "generated_combos": []}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_next_combination(matrix, state, category):
    """Selectionne la prochaine combinaison unique de la matrice.
    Utilise un systeme d'index rotatif pour parcourir toutes les combinaisons
    avant de recommencer, garantissant une variete maximale."""
    data = matrix[category]
    indices = state["last_indices"][category]

    comp_idx = indices["competence"]
    sit_idx = indices["situation"]
    prob_idx = indices["problematique"]

    competence = data["competences"][comp_idx % len(data["competences"])]
    situation = data["situations"][sit_idx % len(data["situations"])]
    problematique = data["problematiques"][prob_idx % len(data["problematiques"])]

    # Avancer les index de maniere desynchronisee pour maximiser la diversite
    indices["competence"] = (comp_idx + 1) % len(data["competences"])
    indices["situation"] = (sit_idx + 3) % len(data["situations"])  # pas de 3
    indices["problematique"] = (prob_idx + 7) % len(data["problematiques"])  # pas de 7 (premier)

    # Aussi utiliser la date comme graine de diversite supplementaire
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_hash = int(hashlib.md5(f"{today}-{category}".encode()).hexdigest()[:8], 16)
    indices["situation"] = (indices["situation"] + date_hash) % len(data["situations"])

    state["last_indices"][category] = indices

    combo_key = f"{category}:{competence}:{situation}:{problematique}"
    state["generated_combos"].append(combo_key)
    # Garder seulement les 1000 derniers pour ne pas exploser le fichier
    if len(state["generated_combos"]) > 1000:
        state["generated_combos"] = state["generated_combos"][-500:]

    return competence, situation, problematique


def call_mammouth(messages, model=MODEL_ARTICLE, max_tokens=3000, temperature=0.85):
    """Appelle l'API Mammouth et retourne le contenu de la reponse."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(API_URL, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        print(f"  ERREUR API ({resp.status_code}): {resp.text[:500]}")
        resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def generate_article(category, competence, situation, problematique):
    """Genere un article complet via l'API Mammouth."""
    category_labels = {
        "communication": "Communication",
        "productivite": "Productivite",
        "memorisation": "Memorisation"
    }
    cat_label = category_labels[category]

    system_prompt = f"""Tu es un expert en developpement personnel, specialise en {cat_label.lower()}.
Tu rediges des articles de blog en francais, captivants et pratiques.
Ton style est chaleureux, accessible et motivant. Tu utilises le tutoiement.
Chaque article doit etre unique, concret et immediatement actionnable."""

    user_prompt = f"""Redige un article de blog complet sur le theme de la {cat_label.lower()}.

PARAMETRES DE GENERATION :
- Competence ciblee : {competence}
- Situation de vie : {situation}
- Problematique identifiee : {problematique}

STRUCTURE OBLIGATOIRE DE L'ARTICLE :

1. **HISTOIRE D'INTRODUCTION** (200-300 mots)
   Raconte une courte histoire realiste et immersive mettant en scene un personnage (prenom francais) dans la situation "{situation}".
   Le personnage fait face a un probleme lie a "{problematique}" qui impacte sa capacite de {cat_label.lower()}.
   L'histoire doit etre vivante, avec des dialogues et des details sensoriels.
   Termine l'histoire sur un moment de prise de conscience.

2. **ANALYSE DU PROBLEME** (150-200 mots)
   Nomme clairement la problematique : "{problematique}".
   Explique pourquoi ce probleme est si courant dans le contexte de "{situation}".
   Donne 2-3 signes qui permettent de reconnaitre ce probleme chez soi.
   Mentionne brievement les consequences si on ne traite pas ce probleme.

3. **3 SOLUTIONS CONCRETES** (300-400 mots au total)
   Pour chaque solution :
   - Un titre clair et accrocheur
   - Une explication du principe (2-3 phrases)
   - Un exercice pratique ou une action concrete a faire des aujourd'hui
   - Un exemple d'application dans le contexte de "{situation}"

4. **CONCLUSION MOTIVANTE** (50-100 mots)
   Rappelle que la competence "{competence}" se developpe avec la pratique.
   Termine par une phrase inspirante et un appel a l'action.

IMPORTANT :
- N'utilise PAS de balises markdown pour le titre principal (il sera ajoute automatiquement)
- Utilise ## pour les sous-titres et ### pour les sous-sous-titres
- Le contenu doit faire entre 800 et 1200 mots au total
- Sois concret et pragmatique, evite les generalites creuses"""

    content = call_mammouth(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=MODEL_ARTICLE,
        max_tokens=3000,
        temperature=0.85
    )

    return content


def generate_seo_metadata(category, competence, situation, problematique, article_content):
    """Genere les metadonnees SEO optimisees pour l'article."""
    prompt = f"""A partir de cet article sur la {category}, genere des metadonnees SEO optimisees.

Competence: {competence}
Situation: {situation}
Problematique: {problematique}

Debut de l'article:
{article_content[:500]}

Reponds UNIQUEMENT en JSON valide avec cette structure exacte (pas de texte avant ou apres):
{{
  "title": "Titre SEO accrocheur (55-65 caracteres max)",
  "description": "Meta description engageante (150-160 caracteres max)",
  "slug": "slug-url-en-minuscules-avec-tirets",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

Regles:
- Le titre doit contenir le mot-cle principal et donner envie de cliquer
- La description doit resumer la valeur de l'article et inclure un appel a l'action
- Le slug doit etre court, descriptif et optimise SEO (sans accents, sans caracteres speciaux)
- Les tags doivent etre pertinents pour le referencement (en francais, sans accents)"""

    response = call_mammouth(
        messages=[{"role": "user", "content": prompt}],
        model=MODEL_SEO,
        max_tokens=500,
        temperature=0.3
    )

    # Extraire le JSON de la reponse (meme s'il y a du texte autour)
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())

    # Fallback si le parsing echoue
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "title": f"{competence.title()} : ameliorez votre {category} en {situation}",
        "description": f"Decouvrez comment surmonter {problematique} et ameliorer votre {category}. 3 solutions concretes et actionnables.",
        "slug": f"{category}-{competence.replace(' ', '-')}-{today}",
        "tags": [category, competence.replace(" ", "-"), "developpement-personnel"]
    }


def create_hugo_article(category, seo, article_content, today):
    """Cree le fichier Markdown Hugo avec le front matter."""
    # Nettoyer le slug
    slug = re.sub(r'[^a-z0-9-]', '', seo["slug"].lower().replace(" ", "-"))
    if not slug:
        slug = f"{category}-{today}"

    filename = f"{today}-{slug}.md"
    filepath = BASE_DIR / "content" / category / filename

    # Construire le front matter
    tags_str = ", ".join(f'"{t}"' for t in seo.get("tags", []))

    front_matter = f"""---
title: "{seo['title'].replace('"', "'")}"
date: {today}T08:00:00+01:00
description: "{seo['description'].replace('"', "'")}"
categories: ["{category.title()}"]
tags: [{tags_str}]
slug: "{slug}"
draft: false
---

"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(front_matter + article_content)

    print(f"  -> Article cree : {filepath}")
    return filepath


def main():
    if not API_KEY:
        raise RuntimeError("MAMMOUTH_API_KEY non definie. Ajoutez-la dans les secrets GitHub.")

    print("=" * 60)
    print(f"Generation d'articles - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    matrix = load_matrix()
    state = load_state()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for category in CATEGORIES:
        print(f"\n--- {category.upper()} ---")

        # 1. Selectionner la combinaison
        competence, situation, problematique = get_next_combination(matrix, state, category)
        print(f"  Competence : {competence}")
        print(f"  Situation  : {situation}")
        print(f"  Probleme   : {problematique}")

        # 2. Generer l'article
        print("  Generation de l'article...")
        article_content = generate_article(category, competence, situation, problematique)

        # 3. Generer les metadonnees SEO
        print("  Generation du SEO...")
        seo = generate_seo_metadata(category, competence, situation, problematique, article_content)
        print(f"  Titre SEO  : {seo['title']}")

        # 4. Creer le fichier Hugo
        create_hugo_article(category, seo, article_content, today)

    # Sauvegarder l'etat
    save_state(state)
    print(f"\n{'=' * 60}")
    print("Generation terminee avec succes !")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
