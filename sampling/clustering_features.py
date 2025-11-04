"""
clustering_features.py
----------------------

Extração de features para clustering e busca de "Farthest in Cluster".

Inclui:
- FeatureExtractor: extrai POS TF, NER TF (spaCy en_core_web_sm) e length (nº tokens)
- fuse_features: concatena (embedding CLS || POS TF || NER TF || length)

Notas:
- Usamos listas estáticas ordenadas para POS e NER visando vetor fixo e reprodutibilidade.
- Para NER, compomos um conjunto padrão com as entidades mais comuns do spaCy; se o pipeline
  carregado tiver labels adicionais, elas são unidas e ordenadas para consistência.
- length é log1p(n_tokens) para evitar escala muito discrepante em relação ao embedding.
- Otimizações possíveis: cache de documentos spaCy (se memória permitir) e processamento em batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

try:
    import spacy
except Exception as e:  # pragma: no cover - fallback de import
    spacy = None


DEFAULT_POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"
]

# Conjunto base de entidades comuns do spaCy en_core_web_sm (pode variar com a versão)
DEFAULT_ENT_LABELS = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "ORDINAL", "CARDINAL"
]


@dataclass
class FeatureExtractor:
    nlp_model: str = "en_core_web_sm"

    def __post_init__(self):
        if spacy is None:
            raise ImportError(
                "spaCy não está instalado. Instale com `pip install spacy` e `python -m spacy download en_core_web_sm`."
            )
        try:
            self.nlp = spacy.load(self.nlp_model, disable=["lemmatizer"])  # leve
        except Exception as e:
            raise RuntimeError(
                f"Falha ao carregar spaCy model '{self.nlp_model}'. Baixe com: python -m spacy download {self.nlp_model}"
            ) from e

        # Define vocab fixos e ordenados
        self.pos_vocab = list(DEFAULT_POS_TAGS)

        # Une labels padrão com as labels do pipeline (se disponíveis)
        ent_labels = set(DEFAULT_ENT_LABELS)
        if "ner" in self.nlp.pipe_names:
            try:
                ner_labels = set(self.nlp.get_pipe("ner").labels)
                ent_labels |= ner_labels
            except Exception:
                pass
        self.ent_vocab = sorted(ent_labels)

        # Mapas para índices
        self.pos_index = {p: i for i, p in enumerate(self.pos_vocab)}
        self.ent_index = {e: i for i, e in enumerate(self.ent_vocab)}

    def featurize(self, text: str) -> Dict[str, np.ndarray]:
        """Extrai vetores POS TF, NER TF e length (log1p(n_tokens)).

        Retorna um dicionário com numpy arrays:
            {
                "pos": np.ndarray[|POS|],
                "ent": np.ndarray[|ENT|],
                "length": np.ndarray[1]
            }
        """
        if not isinstance(text, str) or not text.strip():
            # Vetores vazios → zeros
            return {
                "pos": np.zeros(len(self.pos_vocab), dtype=np.float32),
                "ent": np.zeros(len(self.ent_vocab), dtype=np.float32),
                "length": np.zeros(1, dtype=np.float32),
            }
        doc = self.nlp(text)

        pos_vec = np.zeros(len(self.pos_vocab), dtype=np.float32)
        ent_vec = np.zeros(len(self.ent_vocab), dtype=np.float32)

        token_count = 0
        for tok in doc:
            if tok.is_space:
                continue
            token_count += 1
            p = tok.pos_
            if p in self.pos_index:
                pos_vec[self.pos_index[p]] += 1.0

        # Normaliza POS por número de tokens (evita dependência forte do comprimento)
        if token_count > 0:
            pos_vec /= float(token_count)

        # Frequência de entidades por documento (normalizada pelo total de entidades se >0)
        ent_count = 0
        for ent in doc.ents:
            if ent.label_ in self.ent_index:
                ent_vec[self.ent_index[ent.label_]] += 1.0
                ent_count += 1
        if ent_count > 0:
            ent_vec /= float(ent_count)

        length_vec = np.array([np.log1p(token_count)], dtype=np.float32)

        return {"pos": pos_vec, "ent": ent_vec, "length": length_vec}


def fuse_features(
    emb: np.ndarray,
    pos: np.ndarray,
    ent: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """Concatena (embedding || POS TF || NER TF || length) → vetor final.

    Args:
        emb: vetor do CLS (ex.: DistilRoBERTa hidden_size)
        pos: vetor POS TF
        ent: vetor NER TF
        length: vetor [log1p(n_tokens)]

    Returns:
        np.ndarray concatenado
    """
    if emb.ndim != 1:
        emb = emb.reshape(-1)
    return np.concatenate([emb.astype(np.float32), pos.astype(np.float32), ent.astype(np.float32), length.astype(np.float32)])

