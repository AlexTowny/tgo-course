"""Лабораторная 2. Задание: assignments/02/README.md в репозитории курса.

Запуск: python lab02/run.py, для CI python lab02/run.py --fast.
"""
import argparse
from pathlib import Path
import requests
import re 
from collections import Counter
import json


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"

WORD_RE = re.compile(r'[а-я]+(?:-[а-я]+)*')
END = '</w>'

def load_corpus(url: str):
    response = requests.get(url)
    response.raise_for_status() 
    return response.text


def _normalize(text: str) -> str:
    return text.lower().replace('ё', 'е')


class BPE:
    def __init__(self):
        pass

    def fit(self, text: str, min_merges: int = 1500, min_freq: int = 2):
        words = WORD_RE.findall(text)
        word_freq = Counter(words)

        vocab = {}
        pair_counts = Counter()

        for w, c in word_freq.items():
            if c < min_freq:
                continue
            symbols = list(w) + [END]
            vocab[w] = [symbols, c]
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += c

        merges = []
        for _ in range(min_merges):
            if not pair_counts:
                break
            (a, b), _ = pair_counts.most_common(1)[0]
            merged = a + b
            merges.append((a, b))

            new_vocab = {}
            for w, (symbols, c) in vocab.items():
                # пропускаем слова, где пары нет — экономим O(len)
                has_pair = any(
                    symbols[i] == a and symbols[i + 1] == b
                    for i in range(len(symbols) - 1)
                )
                if not has_pair:
                    new_vocab[w] = [symbols, c]
                    continue

                # снимаем старые пары
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] -= c

                # сливаем
                new_symbols, i = [], 0
                while i < len(symbols):
                    if (i < len(symbols) - 1
                            and symbols[i] == a and symbols[i + 1] == b):
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1

                # добавляем новые пары
                for i in range(len(new_symbols) - 1):
                    pair_counts[(new_symbols[i], new_symbols[i + 1])] += c
                new_vocab[w] = [new_symbols, c]

            vocab = new_vocab

        self.merges = merges
        self.word_freq = word_freq
        self.vocab_size = len(vocab.items())
        self.num_merges = min_merges

        return self

    def apply_bpe(self, word: str) -> list:
        symbols = list(word) + [END]
        for a, b in self.merges:
            merged = a + b
            out, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        return symbols

def save_bpe_report(bpe, text, output_path=None, examples=None):
    """
    Считает статистику по результату BPE и сохраняет отчёт в JSON
    в формате, совместимом с artifacts/*.json.

    Args:
        bpe: обученный объект BPE (должен иметь атрибуты merges, vocab_size)
        text: исходный нормализованный текст корпуса
        output_path: путь для сохранения (по умолчанию artifacts/bpe_report.json)
        examples: список слов для примеров токенизации
                  (по умолчанию — топ-10 самых частых слов корпуса)

    Returns:
        dict — сам отчёт (на случай, если нужно что-то ещё напечатать).
    """
    if output_path is None:
        ARTIFACTS.mkdir(exist_ok=True)
        output_path = ARTIFACTS / "bpe.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- статистика по корпусу ---
    chars = 0
    words = 0
    tokens_bpe = 0
    for w in WORD_RE.findall(text):
        chars += len(w)
        words += 1
        tokens_bpe += len(bpe.apply_bpe(w))

    stats = {
        "chars": chars,
        "words": words,
        "tokens_bpe": tokens_bpe,
        "tokens_per_word": round(tokens_bpe / words, 3) if words else 0.0,
        "compression_vs_chars": round(chars / tokens_bpe, 3) if tokens_bpe else 0.0,
    }

    # --- примеры токенизации ---
    if examples is None:
        examples = ("наполеон", "сказала", "неудовольствие")

    examples_tokens = {w: bpe.apply_bpe(w) for w in examples}

    # --- сборка отчёта ---
    report = {
        "vocab_size": bpe.vocab_size,
        "num_merges": len(bpe.merges),
        "merges": bpe.merges,  # tuple'ы json сам сериализует в списки
        "stats": stats,
        "examples": examples_tokens,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()

    fast_fgl = args.fast 

    ARTIFACTS.mkdir(exist_ok=True)

    # load text corpus
    war_and_peace = _normalize(
        load_corpus(CORPUS_URL)
        )

    bpe = BPE()

    bpe.fit(
        text=war_and_peace,
        min_merges=200 if fast_fgl else 1500,
        min_freq=10
        )

    save_bpe_report(
        bpe, 
        war_and_peace
    )

    return 0
    


if __name__ == "__main__":
    main()
