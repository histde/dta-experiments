"""Encode the DTA documents into character-id numpy shards for xLSTM pretraining.

Second step of the pipeline (after build_tokenizer.py):

* the script streams the `text` column of "histde/dta-documents" (no full download
needed)
* encodes every document with the character tokenizer - which carries
  the shared text normalization (text_normalization.py) inside, so the KenLM
  model, the xLSTM model and the scored DDB pages all live in one character
  space
* writes a flat, EOS-delimited character-id stream in raw
  (headerless) numpy shards

No fixed-length packing happens here: train_xlstm.py windows the shards into fixed-length
instances at load time.

Documents are encoded and written as they stream in, so memory stays at one
shard (--save-every-n-chars). The tokenizer is copied into the output
directory together with metadata.json, so train_xlstm.py only needs
--data-dir.

Usage:
    python prepare_xlstm_data.py [--tokenizer dta_xlstm/tokenizer]
                                     [--dataset histde/dta-documents] [--split train]
                                     [--output-dir dta_xlstm/data]
                                     [--limit-docs N]
                                     [--save-every-n-chars 100000000]
                                     [--num-workers N]

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   fully         # fully | mostly | partially | none
    Human-Reviewed: fully         # fully | partially | minimally | none
"""

import argparse
import json
import multiprocessing
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_DATASET = "histde/dta-documents"

_tokenizer = None


def init_worker(tokenizer_dir):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)


def encode_document(row):
    """Return the character ids of one dataset row ([] if unusable)."""
    text = row.get("text")
    return _tokenizer(text, add_special_tokens=False)["input_ids"] if text else []


class NumpyShardWriter:
    """Accumulates a flat character-id stream and flushes raw (headerless) shards."""

    def __init__(self, output_dir, dtype, save_every_n_chars):
        self.output_dir = output_dir
        self.dtype = dtype
        self.save_every_n_chars = save_every_n_chars
        self.ids = []
        self.num_shards = 0

    def add(self, ids):
        self.ids.extend(ids)
        if len(self.ids) >= self.save_every_n_chars:
            self.flush()

    def flush(self):
        if not self.ids:
            return
        path = self.output_dir / f"dta_{self.num_shards:06d}.npy"
        print(f"Writing numpy shard: {path} ({len(self.ids):,} characters)")
        shard = np.memmap(path, mode="w+", dtype=self.dtype, shape=(len(self.ids),))
        shard[:] = self.ids
        shard.flush()
        self.num_shards += 1
        self.ids = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="dta_xlstm/tokenizer", help="output directory of build_tokenizer.py")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="dta_xlstm/data")
    parser.add_argument("--save-every-n-chars", type=int, default=100_000_000)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id, unk_id = tokenizer.eos_token_id, tokenizer.unk_token_id
    dtype = np.uint8 if len(tokenizer) <= 256 else np.uint16
    print(f"Tokenizer: {args.tokenizer} ({len(tokenizer)} tokens, {dtype.__name__} shards)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("dta_*.npy"):
        stale.unlink()
    tokenizer.save_pretrained(str(output_dir))

    dataset = load_dataset(args.dataset, split=args.split, streaming=True).select_columns(["text"])
    if args.limit_docs is not None:
        dataset = dataset.take(args.limit_docs)

    print(f"Encoding {args.dataset} (split {args.split}) ...")
    writer = NumpyShardWriter(output_dir, dtype, args.save_every_n_chars)
    num_documents = total_chars = unk_chars = 0
    with multiprocessing.Pool(processes=args.num_workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        for i, ids in enumerate(pool.imap(encode_document, dataset, chunksize=4), 1):
            if ids:
                num_documents += 1
                total_chars += len(ids)
                unk_chars += ids.count(unk_id)
                writer.add(ids + [eos_id])
            if i % 500 == 0:
                print(f"  {i} rows streamed, {num_documents} usable, {total_chars:,} characters")
    writer.flush()
    if not num_documents:
        raise SystemExit("No usable documents found")
    print(f"{num_documents} usable documents, {total_chars:,} characters, <unk> rate {100 * unk_chars / total_chars:.5f}%")

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "tokenizer": str(args.tokenizer),
        "num_documents": num_documents,
        "num_chars": total_chars + num_documents,  # incl. one <eos> per document
        "unk_rate": unk_chars / total_chars,
        "vocab_size": len(tokenizer),
        "shard_dtype": dtype.__name__,
        "eos_token_id": eos_id,
        "unk_token_id": unk_id,
        "num_shards": writer.num_shards,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done -> {output_dir}/ ({writer.num_shards} shards, tokenizer, metadata.json)")


if __name__ == "__main__":
    main()
