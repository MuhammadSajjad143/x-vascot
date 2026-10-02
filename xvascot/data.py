"""Dataset loading, tokenisation and batching.

The original notebook used ``tensorflow.keras.preprocessing.text.Tokenizer``.
TensorFlow is not installed here (and TF >= 2.16 removed that module anyway), so
``KerasTokenizer`` below reimplements the exact behaviour we depend on:

* ``text_to_word_sequence`` filtering and splitting,
* frequency-ordered vocabulary with a *stable* sort so count ties keep
  first-seen order, indices starting at 1,
* ``texts_to_sequences`` silently dropping tokens whose index >= ``num_words``
  (no OOV token was configured),
* ``pad_sequences`` with ``padding='pre'`` and ``truncating='pre'``.

Verified against the notebook's stored output: 272 distinct tokens learned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, DATASET_I, DATASET_II

# Keras' default filters (note: no apostrophe, no space)
KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


def text_to_word_sequence(text: str, filters: str = KERAS_FILTERS,
                          lower: bool = False, split: str = " ") -> list[str]:
    """Port of keras.preprocessing.text.text_to_word_sequence."""
    if lower:
        text = text.lower()
    translate_map = str.maketrans({c: split for c in filters})
    text = text.translate(translate_map)
    return [t for t in text.split(split) if t]


class KerasTokenizer:
    """Drop-in replacement for the subset of Keras' Tokenizer the pipeline uses."""

    def __init__(self, num_words: int | None = None, lower: bool = False,
                 filters: str = KERAS_FILTERS, split: str = " "):
        self.num_words = num_words
        self.lower = lower
        self.filters = filters
        self.split = split
        self.word_counts: dict[str, int] = {}
        self.word_index: dict[str, int] = {}

    def fit_on_texts(self, texts) -> "KerasTokenizer":
        for text in texts:
            for w in text_to_word_sequence(str(text), self.filters, self.lower, self.split):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
        # Stable sort by count descending -> ties keep first-seen (insertion) order,
        # matching Keras, which sorts the same list the same way.
        wcounts = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        self.word_index = {w: i for i, (w, _) in enumerate(wcounts, start=1)}
        return self

    def texts_to_sequences(self, texts) -> list[list[int]]:
        out = []
        for text in texts:
            vect = []
            for w in text_to_word_sequence(str(text), self.filters, self.lower, self.split):
                i = self.word_index.get(w)
                if i is None:
                    continue
                if self.num_words and i >= self.num_words:
                    continue          # no oov_token configured -> token is dropped
                vect.append(i)
            out.append(vect)
        return out

    @property
    def index_word(self) -> dict[int, str]:
        return {i: w for w, i in self.word_index.items()}


def pad_sequences(sequences, maxlen: int, value: int = 0,
                  padding: str = "pre", truncating: str = "pre",
                  dtype=np.int64) -> np.ndarray:
    """Port of keras.preprocessing.sequence.pad_sequences (the options we use)."""
    x = np.full((len(sequences), maxlen), value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        trunc = s[-maxlen:] if truncating == "pre" else s[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == "pre":
            x[idx, -len(trunc):] = trunc
        else:
            x[idx, :len(trunc)] = trunc
    return x


# ── Dataset I ────────────────────────────────────────────────────────────────
def load_dataset_i(path=DATASET_I) -> pd.DataFrame:
    """Load Dataset I and map the one-hot Result column to a binary label.

    '1 0 0 0' -> 0 (non-vulnerable), '0 0 0 1' -> 1 (greedy). Rows carrying any
    other class are dropped, matching the notebook.
    """
    df = pd.read_excel(path, sheet_name="Opcode-Result")
    category_map = {"1 0 0 0": 0, "0 1 0 0": 1, "0 0 1 0": 2, "0 0 0 1": 3}
    df = df.copy()
    df["LABEL"] = df["Result"].astype(str).str.strip().map(category_map)
    df["LABEL"] = df["LABEL"].apply(lambda x: 0 if x == 0 else (1 if x == 3 else None))
    df = df.dropna(subset=["LABEL"])
    df["LABEL"] = df["LABEL"].astype(int)
    return df


def load_dataset_ii(path=DATASET_II) -> pd.DataFrame:
    """Load Dataset II-r (unverified 2017 contracts).

    Columns are Adress / Bytecode / Type / Result. ``Type`` is one of
    Suicidal / Greedy / Prodigal; we keep greedy as the positive class.
    """
    df = pd.read_excel(path, sheet_name="Last")
    out = pd.DataFrame({
        "OPCODE": df["Bytecode"].astype(str).str.strip(),
        "TYPE":   df["Type"].astype(str).str.strip().str.lower(),
    })
    out["LABEL"] = out["TYPE"].map({"greedy": 1, "suicidal": 0, "prodigal": 0})
    out = out.dropna(subset=["LABEL"])
    out["LABEL"] = out["LABEL"].astype(int)
    return out


# ── Datasets II and II-r (base paper Table 2), from ``Opcode 16469.xls`` ─────
# The sheet ``Data`` is Dataset II (8,006 safe / 1,207 suicidal / 1,460 prodigal /
# 5,782 greedy); ``Greedy and Non-Vulnerable`` is Dataset II-r (8,006 / 5,782).
# ``Opcode 13801.xlsx`` is NOT Dataset II-r: it holds suicidal/greedy/prodigal
# rows only, with no non-vulnerable contracts.
II_CLASSES = ["non-vulnerable", "suicidal", "prodigal", "greedy"]


def load_dataset_ii_xls(binary: bool, path=None) -> pd.DataFrame:
    """Dataset II-r (binary: safe 0 / greedy 1) or Dataset II (4-class, the
    base paper's one-hot order: safe 0, suicidal 1, prodigal 2, greedy 3)."""
    from .config import DATASET_II_XLS
    sheet = "Greedy and Non-Vulnerable" if binary else "Data"
    df = pd.read_excel(path or DATASET_II_XLS, sheet_name=sheet)
    df = df.rename(columns={"ADDRESS": "Address"})
    df["OPCODE"] = df["OPCODE"].astype(str)
    cmap = {"1 0 0 0": 0, "0 1 0 0": 1, "0 0 1 0": 2, "0 0 0 1": 3}
    lab = df["CATEGORY"].astype(str).str.strip().map(cmap)
    if binary:
        lab = lab.map({0: 0, 3: 1})
    df["LABEL"] = lab
    df = df.dropna(subset=["LABEL"]).reset_index(drop=True)
    df["LABEL"] = df["LABEL"].astype(int)
    return df


def dedup_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per distinct bytecode (Dataset II: 16,455 rows, 2,628 distinct)."""
    return df.drop_duplicates(subset="OPCODE", keep="first").reset_index(drop=True)


def disassemble_tokens(tokens: list[str]) -> list[str]:
    """Keep only real instructions, dropping PUSH immediate data.

    EVM PUSH1..PUSH32 (0x60-0x7f) are each followed by 1..32 bytes of operand.
    The dataset stores the raw byte stream, so reading it one byte at a time
    treats every operand as an instruction. On Dataset I that mislabels 43.6% of
    the stream, and turns the ``ff`` bytes of address masks such as
    ``0xffff...ff`` into phantom SELFDESTRUCT opcodes - 223,343 of them across
    300 contracts, of which 13 are real.
    """
    out, i = [], 0
    while i < len(tokens):
        try:
            b = int(tokens[i], 16)
        except (ValueError, TypeError):
            i += 1
            continue
        out.append(tokens[i])
        i += (1 + (b - 0x5f)) if 0x60 <= b <= 0x7f else 1
    return out


def disassemble_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` whose OPCODE column holds instructions only."""
    df = df.copy()
    df["OPCODE"] = [
        " ".join(disassemble_tokens(str(s).split())) for s in df["OPCODE"].values
    ]
    return df


def build_corpus(df: pd.DataFrame, tokenizer: KerasTokenizer,
                 truncate_to: int | None):
    """Tokenise a dataframe into model inputs.

    Returns ``(X, y, lengths)`` where ``X`` is an ``(N, maxlen)`` int array in
    ``repro`` mode (``truncate_to`` set), or a list of variable-length arrays in
    ``fixed`` mode (``truncate_to=None``).
    """
    seqs = tokenizer.texts_to_sequences(df["OPCODE"].values)
    lengths = np.array([len(s) for s in seqs])
    if truncate_to is not None:
        X = pad_sequences(seqs, maxlen=truncate_to)
    else:
        X = [np.asarray(s, dtype=np.int64) for s in seqs]
    y = df["LABEL"].to_numpy().astype(np.int64)
    return X, y, lengths


def make_tokenizer(df: pd.DataFrame, num_words: int = CONFIG["num_words"]) -> KerasTokenizer:
    return KerasTokenizer(num_words=num_words, lower=False).fit_on_texts(df["OPCODE"].values)
