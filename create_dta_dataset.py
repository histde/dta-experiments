"""Convert a DTA TCF dump into the `documents` config of a HF dataset (Parquet).

One row per DTA work, built from the union of the dump's two variants:

  * simple/  - CMDI metadata + the layout-faithful historical text (<text> layer);
  * full/    - CMDI metadata + annotation layers (tokens, sentences, orthography).

Text columns:
  * text            - historical transcription. From simple/ where available
                      ("layout" in text_source); for works only present in
                      full/ it is reconstructed by space-joining the token
                      layer ("tokens" in text_source).
  * text_normalized - modern orthography: the token layer with the CAB
                      orthography corrections (operation="replace") applied,
                      space-joined. Null for works only present in simple/.

Metadata is taken from the embedded TEI header, deliberately un-flattened:
authors as a list of structs incl. GND ids (empty for anonymous works),
original publication date raw plus a parsed year, all four genre schemes as
separate nullable columns, per-work license URL (licenses VARY across the
corpus - CC BY-SA 2.0/3.0/4.0, CC0, CC BY-NC, ... - so this column matters),
and the DTA extent measures (images/tokens/types/characters) as given.

Usage:
    python create_dta_dataset.py [--dta-dir .../dta_komplett_2026-02-10]
                                 [--output-dir dta_documents]
                                 [--rows-per-shard 1000] [--limit-docs N]
                                 [--num-workers N]

Dataset statistics (license/temporal/genre tables) live in dta_dataset_stats.py.

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   fully         # fully | mostly | partially | none
    Human-Reviewed: none          # fully | partially | minimally | none
"""

import argparse
import multiprocessing
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

DEFAULT_DTA_DIR = "/home/stefan/Repositories/histde-bench/dta_komplett_2026-02-10"
CMDP = "{http://www.clarin.eu/cmd/1/profiles/clarin.eu:cr1:p_1562754657370}"
TC = "{http://www.dspin.de/data/textcorpus}"

SCHEMA = {
    "id": pl.String,
    "dta_id": pl.String,
    "urn": pl.String,
    "url": pl.String,
    "title": pl.String,
    "subtitle": pl.String,
    "volume": pl.String,
    "part": pl.String,
    "authors": pl.List(pl.Struct({"surname": pl.String, "forename": pl.String, "gnd": pl.String})),
    "date_published": pl.String,
    "year": pl.Int64,
    "place_of_publication": pl.String,
    "publisher": pl.String,
    "edition": pl.String,
    "num_pages": pl.Int64,
    "repository": pl.String,
    "shelfmark": pl.String,
    "typeface": pl.String,
    "bibl": pl.String,
    "genre_dtamain": pl.String,
    "genre_dtasub": pl.String,
    "genre_dwds1main": pl.String,
    "genre_dwds1sub": pl.String,
    "dta_corpus_labels": pl.List(pl.String),
    "language": pl.String,
    "language_note": pl.String,
    "license": pl.String,
    "license_family": pl.String,
    "num_images": pl.Int64,
    "num_tokens": pl.Int64,
    "num_types": pl.Int64,
    "num_characters": pl.Int64,
    "num_sentences": pl.Int64,
    "text": pl.String,
    "text_source": pl.String,
    "text_normalized": pl.String,
}

_dta_dir = None  # set per worker via init_worker


def license_family(url):
    """Normalized license identifier for the raw `license` URL (which stays as-is).

    Collapses http/https, `/deed.de`, trailing-slash and country variants into
    one identifier per license+version, e.g. `cc-by-sa-4.0`. NOTE: only
    `cc0-1.0` is a public-domain dedication; `cc-by-*` licenses carry binding
    conditions (attribution, share-alike, or non-commercial).
    """
    if not url:
        return None
    lowered = url.lower()
    match = re.search(r"creativecommons\.org/(?:licenses/(by[a-z-]*)|(publicdomain)/zero)/(\d\.\d)", lowered)
    if match:
        return f"cc0-{match.group(3)}" if match.group(2) else f"cc-{match.group(1)}-{match.group(3)}"
    if "gutenberg" in lowered:
        return "gutenberg"
    if "rightsstatements.org/vocab/noc-nc" in lowered:
        return "noc-nc-1.0"
    if "out-of-copyright-non-commercial" in lowered:
        return "out-of-copyright-nc"
    if "deutsche-digitale-bibliothek.de/lizenzen/nug-kkn" in lowered:
        return "nug-kkn"
    if "mdz.bib-bvb.de" in lowered:
        return "mdz-copyright"
    return "other"


def text_of(element):
    return element.text.strip() if element is not None and element.text and element.text.strip() else None


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_person(person_element):
    name = person_element.find(f"{CMDP}persName")
    if name is None:
        return None
    gnd = name.find(f"{CMDP}idno/{CMDP}idno[@type='PND']")
    return {
        "surname": text_of(name.find(f"{CMDP}surname")),
        "forename": text_of(name.find(f"{CMDP}forename")),
        "gnd": text_of(gnd),
    }


def parse_metadata(root):
    """Extract the documents-config metadata from the embedded TEI header."""
    header = root.find(f".//{CMDP}teiHeader")
    file_desc = header.find(f"{CMDP}fileDesc")
    publication = file_desc.find(f"{CMDP}publicationStmt")
    bibl_full = file_desc.find(f"{CMDP}sourceDesc/{CMDP}biblFull")
    ms_desc = file_desc.find(f"{CMDP}sourceDesc/{CMDP}msDesc")
    profile = header.find(f"{CMDP}profileDesc")

    def idno(type_name):
        return text_of(publication.find(f"{CMDP}idno/{CMDP}idno[@type='{type_name}']"))

    titles = {}
    for title in bibl_full.findall(f"{CMDP}titleStmt/{CMDP}title"):
        title_type = title.get("type") or "main"
        if text_of(title):
            titles.setdefault(title_type, []).append(text_of(title))

    date_element = bibl_full.find(f"{CMDP}publicationStmt/{CMDP}date")
    date_published = text_of(date_element)
    year_match = re.search(r"\b(\d{4})\b", date_published or "")

    publisher_names = [
        text_of(name)
        for publisher in bibl_full.findall(f"{CMDP}publicationStmt/{CMDP}publisher")
        for name in publisher
        if text_of(name)
    ]

    edition_element = bibl_full.find(f"{CMDP}editionStmt/{CMDP}edition")
    licence = publication.find(f"{CMDP}availability/{CMDP}licence")

    genres = {}
    corpus_labels = []
    for class_code in profile.findall(f"{CMDP}textClass/{CMDP}classCode"):
        scheme = (class_code.get("scheme") or "").rsplit("#", 1)[-1]
        if scheme == "DTACorpus":
            if text_of(class_code):
                corpus_labels.append(text_of(class_code))
        elif scheme in ("dtamain", "dtasub", "dwds1main", "dwds1sub"):
            genres.setdefault(scheme, text_of(class_code))

    languages = profile.findall(f"{CMDP}langUsage/{CMDP}language")

    def measure(scope, type_name):
        return to_int(text_of(scope.find(f"{CMDP}extent/{CMDP}measure[@type='{type_name}']")))

    return {
        "id": idno("DTADirName"),
        "dta_id": idno("DTAID"),
        "urn": idno("URN"),
        "url": idno("URLWeb"),
        "title": "\n".join(titles.get("main", [])) or None,
        "subtitle": "\n".join(titles.get("sub", [])) or None,
        "volume": "\n".join(titles.get("volume", [])) or None,
        "part": "\n".join(titles.get("part", [])) or None,
        "authors": [p for p in (parse_person(a) for a in bibl_full.findall(f"{CMDP}titleStmt/{CMDP}author")) if p],
        "date_published": date_published,
        "year": int(year_match.group(1)) if year_match else None,
        "place_of_publication": text_of(bibl_full.find(f"{CMDP}publicationStmt/{CMDP}pubPlace")),
        "publisher": "\n".join(publisher_names) or None,
        "edition": text_of(edition_element) or (edition_element.get("n") if edition_element is not None else None),
        "num_pages": measure(bibl_full, "pages"),
        "repository": text_of(ms_desc.find(f"{CMDP}msIdentifier/{CMDP}repository")) if ms_desc is not None else None,
        "shelfmark": text_of(ms_desc.find(f"{CMDP}msIdentifier/{CMDP}idno/{CMDP}idno[@type='shelfmark']"))
        if ms_desc is not None else None,
        "typeface": text_of(ms_desc.find(f"{CMDP}physDesc/{CMDP}typeDesc/{CMDP}p")) if ms_desc is not None else None,
        "bibl": text_of(file_desc.find(f"{CMDP}sourceDesc/{CMDP}bibl")),
        "genre_dtamain": genres.get("dtamain"),
        "genre_dtasub": genres.get("dtasub"),
        "genre_dwds1main": genres.get("dwds1main"),
        "genre_dwds1sub": genres.get("dwds1sub"),
        "dta_corpus_labels": corpus_labels,
        "language": languages[0].get("ident") if languages else None,
        "language_note": text_of(languages[0]) if languages else None,
        "license": licence.get("target") if licence is not None else None,
        "license_family": license_family(licence.get("target")) if licence is not None else None,
        "num_images": measure(file_desc, "images"),
        "num_tokens": measure(file_desc, "tokens"),
        "num_types": measure(file_desc, "types"),
        "num_characters": measure(file_desc, "characters"),
    }


def parse_full_layers(root):
    """Token list, normalized token list and sentence count from a full/ file."""
    corpus = root.find(f"{TC}TextCorpus")
    tokens = [(t.get("ID"), t.text or "") for t in corpus.findall(f"{TC}tokens/{TC}token")]
    corrections = {}
    skip_ids = set()
    orthography = corpus.find(f"{TC}orthography")
    if orthography is not None:
        for correction in orthography.findall(f"{TC}correction"):
            if correction.get("operation") != "replace":
                continue
            token_ids = (correction.get("tokenIDs") or "").split()
            if token_ids:
                corrections[token_ids[0]] = correction.text or ""
                skip_ids.update(token_ids[1:])  # multi-token corrections replace a span
    historical = [form for _, form in tokens]
    normalized = [corrections.get(tid, form) for tid, form in tokens if tid not in skip_ids]
    sentences = corpus.find(f"{TC}sentences")
    num_sentences = len(sentences.findall(f"{TC}sentence")) if sentences is not None else None
    return historical, normalized, num_sentences


def build_row(document_id):
    simple_file = _dta_dir / "simple" / f"{document_id}.tcf.xml"
    full_file = _dta_dir / "full" / f"{document_id}.tcf.xml"

    try:
        metadata_root = ET.parse(simple_file if simple_file.exists() else full_file).getroot()
        row = parse_metadata(metadata_root)
    except Exception as ex:
        print(f"WARNING - skipping {document_id}: {type(ex).__name__}: {ex}")
        return None
    row["id"] = row["id"] or document_id

    text = None
    if simple_file.exists():
        text_element = metadata_root.find(f"{TC}TextCorpus/{TC}text")
        text = text_of(text_element)

    row["text_normalized"] = None
    row["num_sentences"] = None
    if full_file.exists():
        try:
            full_root = metadata_root if not simple_file.exists() else ET.parse(full_file).getroot()
            historical, normalized, num_sentences = parse_full_layers(full_root)
            row["text_normalized"] = " ".join(normalized) or None
            row["num_sentences"] = num_sentences
            if text is None:
                text = " ".join(historical) or None
                row["text_source"] = "tokens"
        except Exception as ex:
            print(f"WARNING - no annotation layers for {document_id}: {type(ex).__name__}: {ex}")

    row["text"] = text
    row.setdefault("text_source", "layout")
    return row


def init_worker(dta_dir):
    global _dta_dir
    _dta_dir = Path(dta_dir)


def write_shard(rows, output_dir, shard_index):
    df = pl.from_dicts(rows, schema=SCHEMA)
    shard_file = output_dir / f"train-{shard_index:05d}.parquet"
    temp_file = shard_file.with_name(shard_file.name + ".tmp")
    df.write_parquet(temp_file, compression="zstd")
    temp_file.rename(shard_file)
    print(f"Wrote {shard_file} ({df.height} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dta-dir", default=DEFAULT_DTA_DIR)
    parser.add_argument("--output-dir", default="dta_documents")
    parser.add_argument("--rows-per-shard", type=int, default=1000)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    dta_dir = Path(args.dta_dir)
    document_ids = sorted(
        {f.name.removesuffix(".tcf.xml") for variant in ("simple", "full") for f in (dta_dir / variant).glob("*.tcf.xml")}
    )
    if not document_ids:
        raise SystemExit(f"No *.tcf.xml files found under {dta_dir}/simple or {dta_dir}/full")
    if args.limit_docs is not None:
        document_ids = document_ids[: args.limit_docs]
    print(f"{len(document_ids)} documents (union of simple/ and full/)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("train-*.parquet"):
        stale.unlink()

    buffer, shard_index, row_counter = [], 0, 0
    with multiprocessing.Pool(processes=args.num_workers, initializer=init_worker, initargs=(str(dta_dir),)) as pool:
        for row in pool.imap(build_row, document_ids, chunksize=4):
            if row is None:
                continue
            buffer.append(row)
            row_counter += 1
            if len(buffer) >= args.rows_per_shard:
                write_shard(buffer, output_dir, shard_index)
                buffer, shard_index = [], shard_index + 1
    if buffer:
        write_shard(buffer, output_dir, shard_index)
        shard_index += 1

    # Verification
    df = pl.scan_parquet(output_dir / "train-*.parquet").collect()
    assert df.height == row_counter, f"Shards hold {df.height:,} rows, expected {row_counter:,}!"
    print(f"\nDone: {row_counter:,} documents in {shard_index} Parquet shards -> {output_dir}/")
    print(f"Statistics: python dta_dataset_stats.py --dataset-dir {output_dir}")


if __name__ == "__main__":
    main()
