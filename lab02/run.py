"""Лабораторная 2. Задание: assignments/02/README.md в репозитории курса.

Запуск: python lab02/run.py, для CI python lab02/run.py --fast.
"""
import argparse
import csv
import gc
from pathlib import Path
import requests
import re
from collections import Counter
import json

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"

WORD_RE = re.compile(r'[а-я]+(?:-[а-я]+)*')
END = '</w>'

# ---------------------------------------------------------------- константы частей 2-4

QWEN_MODEL = "Qwen/Qwen3-0.6B-Base"
QWEN_TOKENIZER = QWEN_MODEL
E5_MODEL = "intfloat/multilingual-e5-small"
CONTEXT_TOKENS = 4096

EN_PARAGRAPH = (
    "The prince had been talking for almost an hour, and the guests were "
    "growing tired. Anna looked at the clock, then at the window, then at "
    "the clock again. Nobody dared to leave before the host, and the host "
    "did not seem to notice the time at all. A servant brought tea, and the "
    "conversation turned, as it always did in that house, to the war and to "
    "the emperor."
)

ATTENTION_SENTENCE = "Врач осмотрел пациента и сказал, что его анализы готовы."

DILUTION_INSTRUCTION = (
    "Отвечай только по-русски и в конце каждого ответа ставь слово КОНЕЦ."
)

QUERIES = [
    (
        "Хозяйка вечера несколько дней болела гриппом и утром разослала приглашения",
        "кашляла несколько дней",
    ),
    (
        "Отец говорит, что оба его сына дураки, только один спокойный, а другой беспокойный",
        "покойный дурак",
    ),
    (
        "Богатый и скупой старый князь живет в деревне, его прозвали прусским королем",
        "прозванный прусским королем",
    ),
    (
        "Гости по очереди здоровались с никому не нужной тетушкой",
        "никому не интересной и не нужной тетушки",
    ),
    (
        "Брат удивительно похож на сестру-красавицу, но при этом очень некрасив",
        "поразительно дурен собой",
    ),
    (
        "В гостиную входит невысокий молодой князь с усталым и скучающим взглядом",
        "усталого, скучающего взгляда",
    ),
    (
        "Обедневшая княгиня приехала просить, чтобы ее сына перевели в гвардию",
        "выхлопотать определение в гвардию",
    ),
    (
        "Спор о революции: народ отдал Бонапарту власть, чтобы избавиться от Бурбонов",
        "избавил его от Бурбонов",
    ),
    (
        "Неуклюжий толстый гость по рассеянности взял чужую генеральскую шляпу",
        "треугольную шляпу с генеральским плюмажем",
    ),
    (
        "Мальчика десяти лет отправили за границу с гувернером-аббатом",
        "гувернером-аббатом за границу",
    ),
    (
        "Совет другу: не женись, пока не сделал всего, что мог",
        "не женись до тех пор",
    ),
    (
        "Светлая петербургская ночь, второй час, он едет домой и не может заснуть",
        "бессумрачная ночь",
    ),
]

_QWEN_TOKENIZER_CACHE = None


def _save_json(name: str, data) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def get_qwen_tokenizer():
    """Токенизатор Qwen3, загружается один раз на весь запуск."""
    global _QWEN_TOKENIZER_CACHE
    if _QWEN_TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer

        _QWEN_TOKENIZER_CACHE = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)
    return _QWEN_TOKENIZER_CACHE


def _load_auto_model(name: str, **kwargs):
    """AutoModel в float32; dtype/torch_dtype для совместимости версий transformers."""
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(name, dtype=torch.float32, **kwargs)
    except TypeError:
        return AutoModel.from_pretrained(name, torch_dtype=torch.float32, **kwargs)


def count_tokens(text: str, tokenizer=None) -> int:
    """Число токенов Qwen3 в тексте, без служебных токенов."""
    tokenizer = tokenizer or get_qwen_tokenizer()
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def load_corpus(url: str):
    response = requests.get(url, timeout=60)
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
    token_cache = {}
    for w in WORD_RE.findall(text):
        chars += len(w)
        words += 1
        if w not in token_cache:
            token_cache[w] = bpe.apply_bpe(w)
        tokens_bpe += len(token_cache[w])

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


# ---------------------------------------------------------------- часть 2: токенизатор и бюджет контекста

def build_tokens_per_word(corpus: str, tokenizer=None) -> dict:
    """Сравнивает стоимость русского и английского текста в токенах Qwen3."""
    tokenizer = tokenizer or get_qwen_tokenizer()

    # Слова считаем по нормализованному тексту, как в части 1, и берем 500:600.
    corpus_words = WORD_RE.findall(_normalize(corpus))
    ru_text = " ".join(corpus_words[500:600])
    ru_words = len(corpus_words[500:600])

    en_words = len(re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", EN_PARAGRAPH))

    ru_tokens = count_tokens(ru_text, tokenizer)
    en_tokens = count_tokens(EN_PARAGRAPH, tokenizer)

    def language_stats(words: int, tokens: int) -> dict:
        tokens_per_word = round(tokens / words, 3)
        return {
            "words": words,
            "tokens": tokens,
            "tokens_per_word": tokens_per_word,
            "words_in_context": int(CONTEXT_TOKENS / tokens_per_word),
        }

    example_ids = tokenizer(ru_text, add_special_tokens=False)["input_ids"]

    return {
        "tokenizer": QWEN_TOKENIZER,
        "vocab_size": len(tokenizer),
        "context_tokens": CONTEXT_TOKENS,
        "ru": language_stats(ru_words, ru_tokens),
        "en": language_stats(en_words, en_tokens),
        "example_ru": [tokenizer.decode([i]) for i in example_ids[:20]],
    }


# ---------------------------------------------------------------- часть 3: векторы и поиск

def select_paragraphs(corpus: str, limit: int) -> list:
    """Абзацы по переводам строки, 40-120 слов, первые limit штук."""
    paragraphs = []
    for line in corpus.splitlines():
        words = WORD_RE.findall(line.lower())
        if 40 <= len(words) <= 120:
            paragraphs.append(line)  # храним как есть, без strip
            if len(paragraphs) >= limit:
                break
    return paragraphs


def _batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def compute_qwen_vectors(texts, tokenizer, model, batch_size: int = 8):
    """static (входные эмбеддинги) и contextual (последний слой) векторы Qwen3."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    static_parts = []
    contextual_parts = []

    for batch in _batches(texts, batch_size):
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
            mask = encoded["attention_mask"]
            embeddings = model.get_input_embeddings()(encoded["input_ids"])
            static = _masked_mean(embeddings, mask)
            contextual = _masked_mean(outputs.hidden_states[-1], mask)

        static_parts.append(static.float().cpu().numpy())
        contextual_parts.append(contextual.float().cpu().numpy())

    return (
        np.concatenate(static_parts, axis=0),
        np.concatenate(contextual_parts, axis=0),
    )


def compute_e5_vectors(texts, prefix: str, tokenizer, model, batch_size: int = 32):
    """Среднее по маске выходов multilingual-e5-small; префикс query/passage."""
    parts = []
    for batch in _batches(texts, batch_size):
        encoded = tokenizer(
            [prefix + text for text in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
            pooled = _masked_mean(hidden, encoded["attention_mask"])
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        parts.append(pooled.float().cpu().numpy())
    return np.concatenate(parts, axis=0)


def cosine_top_k(query_vectors, paragraph_vectors, k: int = 5):
    """Для каждого запроса — номера k ближайших абзацев по косинусу."""
    query = query_vectors / np.linalg.norm(
        query_vectors, axis=1, keepdims=True
    ).clip(min=1e-12)
    paragraphs = paragraph_vectors / np.linalg.norm(
        paragraph_vectors, axis=1, keepdims=True
    ).clip(min=1e-12)
    scores = query @ paragraphs.T
    return [np.argsort(-row)[:k].tolist() for row in scores]


def run_search(corpus: str, tokenizer, fast: bool) -> dict:
    from transformers import AutoTokenizer

    limit = 60 if fast else 500
    paragraphs = select_paragraphs(corpus, limit)
    query_texts = [query for query, _ in QUERIES]

    print(f"[3] абзацев: {len(paragraphs)}", flush=True)

    qwen_model = _load_auto_model(QWEN_MODEL)
    qwen_model.eval()
    static_paragraphs, contextual_paragraphs = compute_qwen_vectors(
        paragraphs, tokenizer, qwen_model
    )
    static_queries, contextual_queries = compute_qwen_vectors(
        query_texts, tokenizer, qwen_model
    )
    del qwen_model
    gc.collect()

    e5_tokenizer = AutoTokenizer.from_pretrained(E5_MODEL)
    e5_model = _load_auto_model(E5_MODEL)
    e5_model.eval()
    e5_paragraphs = compute_e5_vectors(
        paragraphs, "passage: ", e5_tokenizer, e5_model
    )
    e5_queries = compute_e5_vectors(
        query_texts, "query: ", e5_tokenizer, e5_model
    )
    del e5_model
    gc.collect()

    vectors = {
        "static": (static_queries, static_paragraphs),
        "contextual": (contextual_queries, contextual_paragraphs),
        "e5": (e5_queries, e5_paragraphs),
    }
    tops = {
        method: cosine_top_k(query_vecs, paragraph_vecs, k=5)
        for method, (query_vecs, paragraph_vecs) in vectors.items()
    }

    answer_ids = []
    for _, answer_phrase in QUERIES:
        hits = [
            i for i, paragraph in enumerate(paragraphs)
            if answer_phrase in paragraph
        ]
        answer_ids.append(hits[0] if hits else -1)

    queries_report = []
    for qi, ((query, answer_phrase), answer_id) in enumerate(
        zip(QUERIES, answer_ids)
    ):
        queries_report.append({
            "query": query,
            "answer_phrase": answer_phrase,
            "answer_id": answer_id,
            "top": {method: tops[method][qi] for method in vectors},
        })

    recall = {
        method: round(
            sum(answer_id in tops[method][qi]
                for qi, answer_id in enumerate(answer_ids)) / len(QUERIES),
            3,
        )
        for method in vectors
    }

    _save_json("paragraphs.json", [
        {"id": i, "text": paragraph}
        for i, paragraph in enumerate(paragraphs)
    ])
    _save_json("search.json", {
        "k": 5,
        "n_paragraphs": len(paragraphs),
        "queries": queries_report,
        "recall_at_k": recall,
    })

    print(f"[3] recall@5: {recall}", flush=True)
    return {"recall_at_k": recall, "n_paragraphs": len(paragraphs)}


# ---------------------------------------------------------------- часть 4: внимание

def _word_token_positions(sentence: str, offsets, word: str) -> list:
    """Позиции токенов, перекрывающихся со словом word в предложении."""
    match = re.search(
        rf"(?<![а-яё]){re.escape(word)}(?![а-яё])",
        sentence.lower(),
    )
    if match is None:
        raise ValueError(f"в предложении нет слова «{word}»")
    start, end = match.span()
    return [
        i for i, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]


def _save_attention_plot(tokens, weights, pronoun_pos, antecedent_pos, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = []
    for i in range(len(tokens)):
        if i in antecedent_pos:
            colors.append("#d95f02")
        elif i in pronoun_pos:
            colors.append("#1b9e77")
        else:
            colors.append("#9e9e9e")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(len(tokens)), weights, color=colors)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right")
    ax.set_ylabel("attention weight")
    ax.set_title(
        "Qwen3: внимание местоимения «его»; оранжевым — «пациента», зелёным — «его»"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_attention(corpus: str, tokenizer, fast: bool) -> dict:
    model = _load_auto_model(QWEN_MODEL, attn_implementation="eager")
    model.eval()

    # --- 4а: местоимение и антецедент
    encoded = tokenizer(
        ATTENTION_SENTENCE,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"][0].tolist()
    offsets = encoded["offset_mapping"][0].tolist()
    tokens = [tokenizer.decode([i]) for i in token_ids]

    pronoun_pos = _word_token_positions(ATTENTION_SENTENCE, offsets, "его")
    antecedent_pos = _word_token_positions(
        ATTENTION_SENTENCE, offsets, "пациента"
    )
    pronoun_last = pronoun_pos[-1]

    with torch.inference_mode():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_attentions=True,
        )

    best_share = -1.0
    best_layer = 0
    best_head = 0
    for layer, attention in enumerate(outputs.attentions):
        rows = attention[0, :, pronoun_last, :]       # [головы, токены]
        shares = rows[:, antecedent_pos].sum(dim=-1)
        head = int(torch.argmax(shares))
        share = float(shares[head])
        if share > best_share:
            best_share = share
            best_layer = layer
            best_head = head

    weights = (
        outputs.attentions[best_layer][0, best_head, pronoun_last, :]
        .float().cpu().tolist()
    )
    share_on_antecedent = float(sum(weights[i] for i in antecedent_pos))

    attention_report = {
        "sentence": ATTENTION_SENTENCE,
        "tokens": tokens,
        "pronoun_pos": pronoun_pos,
        "antecedent_pos": antecedent_pos,
        "best_layer": best_layer,
        "best_head": best_head,
        "weights": weights,
        "share_on_antecedent": share_on_antecedent,
    }
    _save_json("attention_map.json", attention_report)
    _save_attention_plot(
        tokens,
        weights,
        pronoun_pos,
        antecedent_pos,
        ARTIFACTS / "attention_map.png",
    )
    del outputs
    gc.collect()
    print(
        f"[4a] лучший слой/голова: {best_layer}/{best_head}, "
        f"доля на «пациента»: {share_on_antecedent:.4f}",
        flush=True,
    )

    # --- 4б: размывание инструкции
    lengths = [64, 256, 768] if fast else [128, 512, 1024]
    instruction_ids = tokenizer(
        DILUTION_INSTRUCTION + "\n",
        add_special_tokens=False,
    )["input_ids"]
    instruction_len = len(instruction_ids)

    filler_text = " ".join(corpus.splitlines())[:20000]
    filler_ids = tokenizer(
        filler_text, add_special_tokens=False
    )["input_ids"]
    if len(filler_ids) < max(lengths) - instruction_len:
        raise RuntimeError("в корпусе не хватает текста для наполнителя")

    rows = []
    for context_len in lengths:
        input_ids = instruction_ids + filler_ids[
            : context_len - instruction_len
        ]
        if len(input_ids) != context_len:
            raise RuntimeError("не удалось собрать вход нужной длины")

        with torch.inference_mode():
            outputs = model(
                input_ids=torch.tensor([input_ids]),
                attention_mask=torch.ones(1, context_len, dtype=torch.long),
                output_attentions=True,
            )
            # Средняя по слоям и головам строка внимания последнего токена.
            average_row = torch.stack(
                [attention[0, :, -1, :] for attention in outputs.attentions]
            ).mean(dim=(0, 1))

        sink_share = float(average_row[0])
        share_on_instruction = float(
            average_row[1:instruction_len].sum()
        )
        rows.append({
            "context_len": context_len,
            "instruction_len": instruction_len - 1,
            "share_on_instruction": round(share_on_instruction, 6),
            "sink_share": round(sink_share, 6),
            "uniform_share": round(
                (instruction_len - 1) / context_len, 6
            ),
        })
        del outputs, average_row
        gc.collect()

    with open(
        ARTIFACTS / "dilution.csv", "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "context_len",
                "instruction_len",
                "share_on_instruction",
                "sink_share",
                "uniform_share",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[4b] dilution: {rows}", flush=True)

    del model
    gc.collect()
    return {"attention": attention_report, "dilution": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый режим для CI")
    args = ap.parse_args()

    fast_fgl = args.fast

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(CORPUS_URL)
    war_and_peace = _normalize(corpus)

    # --- часть 1: свой BPE
    bpe = BPE()
    bpe.fit(
        text=war_and_peace,
        min_merges=200 if fast_fgl else 1500,
        min_freq=10,
    )
    bpe_report = save_bpe_report(bpe, war_and_peace)
    print(f"[1] BPE: {bpe_report['stats']}", flush=True)

    # --- часть 2: токенизатор Qwen3
    tokenizer = get_qwen_tokenizer()
    tokens_per_word = build_tokens_per_word(corpus, tokenizer)
    _save_json("tokens_per_word.json", tokens_per_word)
    print(
        f"[2] ru: {tokens_per_word['ru']}, en: {tokens_per_word['en']}",
        flush=True,
    )

    # --- часть 3: векторы и поиск
    run_search(corpus, tokenizer, fast_fgl)

    # --- часть 4: карты внимания и размывание
    run_attention(corpus, tokenizer, fast_fgl)

    return 0
    


if __name__ == "__main__":
    main()
