"""Build the character tokenizer for the DTA xLSTM from the Hugging Face dataset.

First step of the pipeline (run before prepare_dta_xlstm_data.py):

* it streams the `text` column of "histde/dta-documents"
* normalizes it with the shared normalization in `text_normalization.py`,
  so the KenLM model, the xLSTM model and the scored DDB pages all live in one
  character space)
* counts characters
* keeps the most frequent ones up to --vocab-size (default 256, which covers
  >99.99% of DTA characters and lets data shards use uint8), plus <eos> (id 0)
  and <unk> (id 1).

The result is a Hugging Face fast tokenizer (tokenizer.json +
tokenizer_config.json) that carries the normalization inside its `tokenizers`
normalizer pipeline:

* dehyphenation,
* NFC
* lowercasing
* combining-e umlauts
* long s / r rotunda
* quote and dash unification
* digits -> 0
* whitespace collapsing

Every character of the normalized text is one token (<unk> for
characters outside the vocabulary). No special tokens are added
automatically.

`AutoTokenizer.from_pretrained(...)` works without
`trust_remote_code` and training data and scoring can no longer diverge in
normalization. A human-readable vocab.json and tokenizer_stats.json are
written alongside.

Usage:
    python build_tokenizer.py [--dataset histde/dta-documents] [--split train]
                              [--output-dir dta_xlstm/tokenizer]
                              [--vocab-size 256] [--limit-docs N] [--num-workers N]

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   fully         # fully | mostly | partially | none
    Human-Reviewed: fully         # fully | partially | minimally | none
"""

import argparse
import json
import multiprocessing
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from text_normalization import _CHAR_MAP, _COMBINING_E, normalize

DEFAULT_DATASET = "histde/dta-documents"
EOS_TOKEN, EOS_ID = "<eos>", 0
UNK_TOKEN, UNK_ID = "<unk>", 1
NUM_SPECIALS = 2


def build_normalizer():
    """`tokenizers` equivalent of text_normalization.normalize(), same step order."""
    steps = [
        normalizers.Replace("⸗\n", ""),
        normalizers.Replace("-\n", ""),
        normalizers.NFC(),
        # Python's str.lower() applies the Unicode final-sigma rule (word-final
        # Σ -> ς), Rust's Lowercase does not; emulate it before lowercasing.
        normalizers.Replace(Regex(r"(?<=\p{L}\p{M}*)Σ(?!\p{M}*\p{L})"), "ς"),
        normalizers.Lowercase(),
    ]
    steps += [normalizers.Replace(sequence, replacement) for sequence, replacement in _COMBINING_E.items()]
    steps.append(normalizers.Replace("ͤ", ""))
    steps += [normalizers.Replace(chr(source), target) for source, target in _CHAR_MAP.items()]
    steps += [
        normalizers.Replace(Regex(r"\d"), "0"),
        normalizers.Replace(Regex(r"\s+"), " "),
        normalizers.Strip(),
    ]
    return normalizers.Sequence(steps)


def build_tokenizer(vocab):
    tokenizer = Tokenizer(WordLevel(vocab, unk_token=UNK_TOKEN))
    tokenizer.normalizer = build_normalizer()
    # One token per character, whitespace included (Split keeps every piece).
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS_TOKEN,
        bos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
        pad_token=EOS_TOKEN,
        clean_up_tokenization_spaces=False,
        # Never treat literal "<eos>"/"<unk>" in input text as special tokens:
        # scored text must go through the normalizer exactly like normalize().
        split_special_tokens=True,
        model_max_length=1 << 30,
    )


def count_characters(row):
    text = row.get("text")
    return Counter(normalize(text)) if text else Counter()


def stream_rows(args):
    dataset = load_dataset(args.dataset, split=args.split, streaming=True).select_columns(["text"])
    return dataset.take(args.limit_docs) if args.limit_docs is not None else dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="dta_xlstm/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=256,
                        help="total vocabulary incl. <eos>/<unk>; <= 256 keeps data shards uint8")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    print(f"Counting characters in {args.dataset} (split {args.split}) ...")
    counts = Counter()
    num_documents = 0
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for i, document_counts in enumerate(pool.imap(count_characters, stream_rows(args), chunksize=4), 1):
            if document_counts:
                counts.update(document_counts)
                num_documents += 1
            if i % 500 == 0:
                print(f"  {i} rows streamed, {sum(counts.values()):,} characters")
    if not counts:
        raise SystemExit("No usable documents found")

    total_chars = sum(counts.values())
    kept = [char for char, _ in counts.most_common(args.vocab_size - NUM_SPECIALS)]
    vocab = {EOS_TOKEN: EOS_ID, UNK_TOKEN: UNK_ID}
    vocab.update({char: NUM_SPECIALS + i for i, char in enumerate(kept)})
    unk_chars = total_chars - sum(counts[char] for char in kept)
    print(f"{num_documents} usable documents, {total_chars:,} characters, {len(counts)} distinct")
    print(f"Vocabulary: {len(vocab)} entries, <unk> rate {100 * unk_chars / total_chars:.5f}%")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_tokenizer(vocab).save_pretrained(str(output_dir))
    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    stats = {
        "dataset": args.dataset,
        "split": args.split,
        "num_documents": num_documents,
        "num_chars": total_chars,
        "distinct_chars": len(counts),
        "vocab_size": len(vocab),
        "unk_rate": unk_chars / total_chars,
        "eos_token_id": EOS_ID,
        "unk_token_id": UNK_ID,
        "char_counts": {char: counts[char] for char in kept},
        "dropped_chars": {char: count for char, count in counts.most_common() if char not in vocab},
    }
    with open(output_dir / "tokenizer_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Done -> {output_dir}/ (tokenizer.json, tokenizer_config.json, vocab.json, tokenizer_stats.json)")


if __name__ == "__main__":
    main()
