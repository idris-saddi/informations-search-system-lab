import logging
import os

from corpus import corpus_processed, vocabulary, documents
from llm import LLMJudge, benchmark
from models.BM25Model import BM25Model
from models.booleanModel import BooleanModel
from models.languageModel import LanguageModelJM
from models.vectorSpaceModel import VectorSpaceModel


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    _configure_logging()
    # Instanciation
    vsm = VectorSpaceModel(corpus_processed, vocabulary)
    bm25 = BM25Model(corpus_processed)
    lm_jm = LanguageModelJM(corpus_processed)
    boolean = BooleanModel(corpus_processed)

    models = {
        "Vector Space (TF-IDF)": vsm,
        "Probabiliste (BM25)": bm25,
        "Langue (Jelinek-Mercer)": lm_jm,
        "Booleen (AND/OR/NOT)": boolean,
    }

    # Requêtes de benchmark (a enrichir selon votre cours / dataset)
    queries = [
        # "car insurance",
        # "best car insurance",
        # "auto insurance coverage",
        # "michael jackson king of pop",
        # "python data science",
        "machine learning artificial intelligence"
    ]

    judge = LLMJudge()  # utilise Gemini si configure, sinon heuristique + cache
    benchmark(queries, models, documents, top_k=5, judge=judge)


if __name__ == "__main__":
    main()
