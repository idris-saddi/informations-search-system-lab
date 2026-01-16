# Information Search System Lab

Mini-lab de recherche d'information (IR) avec plusieurs modeles + une etude comparative via **LLM-as-a-judge**.

## Contenu

- Corpus jouet (20 docs) dans [corpus.py](corpus.py)
- Modeles dans [models/](models/)
  - Modele vectoriel TF-IDF (cosine)
  - Modele probabiliste BM25
  - Modele de langue (Jelinek–Mercer)
  - Modele booleen (AND/OR/NOT, parentheses)
- evaluation LLM-as-a-judge dans [llm.py](llm.py)

## Installation

### Dependances minimales

```bash
pip install numpy
```

### Optionnel: utiliser Groq comme juge

```bash
pip install groq
```

Creer un fichier `.env` a la racine:

```env
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-8b-instant
```

Si la cle est absente (ou quota depasse) et qu'il n'y a pas de jugement en cache, l'execution echoue (pas de fallback heuristique).

### Optionnel: utiliser Gemini comme juge (recommande)

```bash
pip install google-genai
```

Creer un fichier `.env` a la racine:

```env
GEMINI_API_KEY=...
# Modele (defaut): gemini-flash-lite-latest
GEMINI_MODEL=gemini-flash-lite-latest
```

En mode `auto` (defaut), le juge choisit:
Gemini si configure, sinon Groq si configure, sinon un fallback **heuristique** (local) + cache.

Pour limiter la frequence des appels (rate limit faible):

```env
GEMINI_MIN_DELAY_MS=500
```

## Execution

Lancer l'etude comparative multi-requêtes:

```bash
python main.py
```

### Logs (Groq utilise / erreurs)

Le script ecrit des logs via `logging` (console) pour indiquer:
- Quand l'API Groq est appelee (start/success)
- Si un appel Groq a echoue (error + stacktrace)

Vous pouvez ajuster le niveau:

```bash
set LOG_LEVEL=DEBUG
python main.py
```

Resultats:

- Logs detailles par requête et par modele dans la console
- Rapport JSON ecrit dans `reports/llm_judge_benchmark.json`
- Cache des jugements dans `.cache/llm_judge_cache.json`

## Personnalisation rapide

- Ajouter/editer des documents: [documents](corpus.py)
- Changer les requêtes d'evaluation: [main.py](main.py)
- Ajuster le top-k et le modele Gemini: [llm.py](llm.py)

## Notes methodologiques (LLM-as-a-judge)

- Pour chaque requête, on attribue une pertinence **0/1/2** a **tous** les documents du corpus (petit corpus)
- On calcule ensuite `nDCG@k` avec un `IDCG@k` base sur ces jugements (plus robuste qu'un IDCG sur les seuls docs retournes)
- Le cache permet de rejouer l'etude sans repayer les appels LLM
