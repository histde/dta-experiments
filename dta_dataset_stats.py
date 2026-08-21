"""Print statistics for the DTA `documents` Parquet dataset as GitHub markdown tables.

Reads the shards written by create_dta_dataset.py and prints tables that can be
pasted directly into a dataset card: overview, license distribution, temporal
distribution, text availability, genres, typefaces, languages, DTA corpus
labels, places of publication and most frequent authors.

Usage:
    python dta_dataset_stats.py [--dataset-dir dta_documents] [--bin-size 50]
                                [--top-n 10] [--format github]

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   fully         # fully | mostly | partially | none
    Human-Reviewed: none          # fully | partially | minimally | none
"""

import argparse
from pathlib import Path

import polars as pl
from tabulate import tabulate


def fmt_int(value):
    return f"{value:,}" if value is not None else "n/a"


def fmt_pct(value):
    return f"{value:.1f}%" if value is not None else "n/a"


def print_table(title, rows, headers, table_format):
    print(f"### {title}\n")
    print(tabulate(rows, headers=headers, tablefmt=table_format, disable_numparse=True))
    print()


def overview_rows(df):
    with_full = df["text_normalized"].is_not_null().sum()
    with_authors = (df["authors"].list.len() > 0).sum()
    return [
        ["Documents", fmt_int(df.height)],
        ["Documents with normalized text (full/ layer)", f"{fmt_int(with_full)} ({fmt_pct(100 * with_full / df.height)})"],
        ["Documents with at least one author", f"{fmt_int(with_authors)} ({fmt_pct(100 * with_authors / df.height)})"],
        ["Distinct authors (by GND id)", fmt_int(df["authors"].explode(empty_as_null=True).struct.field("gnd").drop_nulls().n_unique())],
        ["Year range", f"{df['year'].min()}-{df['year'].max()} ({df['year'].null_count()} unknown)"],
        ["Median year", str(int(df["year"].median()))],
        ["Tokens (header measure)", fmt_int(df["num_tokens"].sum())],
        ["Types (header measure)", fmt_int(df["num_types"].sum())],
        ["Characters (header measure)", fmt_int(df["num_characters"].sum())],
        ["Characters in `text` column", fmt_int(df["text"].str.len_chars().sum())],
        ["Characters in `text_normalized` column", fmt_int(df["text_normalized"].str.len_chars().sum())],
        ["Sentences (full/ layer)", fmt_int(df["num_sentences"].sum())],
        ["Page images", fmt_int(df["num_images"].sum())],
        ["Median tokens per document", fmt_int(int(df["num_tokens"].median()))],
        ["Median pages (images) per document", fmt_int(int(df["num_images"].median()))],
    ]


def license_rows(df):
    total_tokens = df["num_tokens"].sum()
    grouped = (
        df.group_by("license_family")
        .agg(
            pl.len().alias("docs"),
            pl.col("num_tokens").sum().alias("tokens"),
            pl.col("license").unique().sort().alias("urls"),
        )
        .sort(["docs", "license_family"], descending=[True, False], nulls_last=True)
    )
    rows = [
        [
            f"`{r['license_family']}`" if r["license_family"] else "unknown",
            fmt_int(r["docs"]),
            fmt_pct(100 * r["docs"] / df.height),
            fmt_int(r["tokens"]),
            fmt_pct(100 * r["tokens"] / total_tokens),
            "<br>".join(r["urls"]) if r["urls"] else "",
        ]
        for r in grouped.to_dicts()
    ]
    rows.append(["**total**", f"**{fmt_int(df.height)}**", "**100%**", f"**{fmt_int(total_tokens)}**", "**100%**", ""])
    return rows


def temporal_rows(df, bin_size):
    total_tokens = df["num_tokens"].sum()
    bins = (
        df.with_columns((pl.col("year") // bin_size * bin_size).alias("bin"))
        .group_by("bin")
        .agg(
            pl.len().alias("docs"),
            pl.col("num_sentences").sum().alias("sentences"),
            pl.col("num_tokens").sum().alias("tokens"),
            pl.col("num_characters").sum().alias("characters"),
        )
        .sort("bin", nulls_last=True)
    )
    rows = []
    for r in bins.to_dicts():
        label = f"{r['bin']}–{r['bin'] + bin_size - 1}" if r["bin"] is not None else "unknown"
        rows.append(
            [label, fmt_int(r["docs"]), fmt_int(r["sentences"]), fmt_int(r["tokens"]), fmt_int(r["characters"]), fmt_pct(100 * r["tokens"] / total_tokens)]
        )
    rows.append(
        [
            "**total**",
            f"**{fmt_int(df.height)}**",
            f"**{fmt_int(df['num_sentences'].sum())}**",
            f"**{fmt_int(total_tokens)}**",
            f"**{fmt_int(df['num_characters'].sum())}**",
            "**100%**",
        ]
    )
    return rows


def distribution_rows(df, column, top_n=None, explode=False, code=False):
    """Docs + tokens + shares per distinct value of `column`; nulls shown as `unknown`."""
    series = df.select(column, "num_tokens")
    if explode:
        series = series.explode(column, empty_as_null=True)
    total_docs, total_tokens = df.height, df["num_tokens"].sum()
    grouped = (
        series.group_by(column)
        .agg(pl.len().alias("docs"), pl.col("num_tokens").sum().alias("tokens"))
        .sort(["docs", column], descending=[True, False], nulls_last=True)
    )
    shown = grouped.head(top_n) if top_n else grouped
    rows = []
    for r in shown.to_dicts():
        value = r[column]
        label = (f"`{value}`" if code else str(value)) if value is not None else "unknown"
        rows.append([label, fmt_int(r["docs"]), fmt_pct(100 * r["docs"] / total_docs), fmt_int(r["tokens"]), fmt_pct(100 * r["tokens"] / total_tokens)])
    if top_n and grouped.height > top_n:
        rest = grouped.slice(top_n)
        rows.append(
            [
                f"*other ({grouped.height - top_n:,} values)*",
                fmt_int(rest["docs"].sum()),
                fmt_pct(100 * rest["docs"].sum() / total_docs),
                fmt_int(rest["tokens"].sum()),
                fmt_pct(100 * rest["tokens"].sum() / total_tokens),
            ]
        )
    return rows


def author_rows(df, top_n):
    authors = (
        df.select("authors", "num_tokens")
        .explode("authors", empty_as_null=True)
        .unnest("authors")
        .filter(pl.col("surname").is_not_null() | pl.col("forename").is_not_null())
        .with_columns(pl.concat_str([pl.col("surname"), pl.col("forename")], separator=", ", ignore_nulls=True).alias("name"))
        .group_by("name", "gnd")
        .agg(pl.len().alias("docs"), pl.col("num_tokens").sum().alias("tokens"))
        .sort(["docs", "tokens", "name"], descending=[True, True, False])
        .head(top_n)
    )
    return [[r["name"], r["gnd"] or "", fmt_int(r["docs"]), fmt_int(r["tokens"])] for r in authors.to_dicts()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="hf-dta-documents", help="Folder with train-*.parquet shards")
    parser.add_argument("--bin-size", type=int, default=50, help="Year bin size for the temporal distribution")
    parser.add_argument("--top-n", type=int, default=10, help="Rows shown in top-N tables")
    parser.add_argument("--format", default="github", help="tabulate table format (github, simple, latex, ...)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    shards = sorted(dataset_dir.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"No train-*.parquet shards found in {dataset_dir}/")
    df = pl.read_parquet(shards)
    fmt = args.format
    shares = ["docs", "docs %", "tokens", "tokens %"]

    print(f"## Dataset statistics\n\nSource: `{dataset_dir}/` ({len(shards)} shards)\n")
    print_table("Overview", overview_rows(df), ["statistic", "value"], fmt)
    print_table("License distribution", license_rows(df), ["license family"] + shares + ["license URLs"], fmt)
    print_table(
        f"Temporal distribution ({args.bin_size}-year bins)",
        temporal_rows(df, args.bin_size),
        ["period", "docs", "sentences", "tokens", "characters", "token share"],
        fmt,
    )
    print_table("Text source", distribution_rows(df, "text_source", code=True), ["text_source"] + shares, fmt)
    print_table("Genre (DTA main)", distribution_rows(df, "genre_dtamain"), ["genre_dtamain"] + shares, fmt)
    print_table(f"Genre (DTA sub, top {args.top_n})", distribution_rows(df, "genre_dtasub", args.top_n), ["genre_dtasub"] + shares, fmt)
    print_table("Genre (DWDS main)", distribution_rows(df, "genre_dwds1main"), ["genre_dwds1main"] + shares, fmt)
    print_table(f"Genre (DWDS sub, top {args.top_n})", distribution_rows(df, "genre_dwds1sub", args.top_n), ["genre_dwds1sub"] + shares, fmt)
    print_table("Typeface", distribution_rows(df, "typeface"), ["typeface"] + shares, fmt)
    print_table("Language", distribution_rows(df, "language", code=True), ["language"] + shares, fmt)
    print_table(
        "DTA corpus labels (a document can carry several)",
        distribution_rows(df, "dta_corpus_labels", explode=True, code=True),
        ["label"] + shares,
        fmt,
    )
    print_table(
        f"Place of publication (top {args.top_n})",
        distribution_rows(df, "place_of_publication", args.top_n),
        ["place"] + shares,
        fmt,
    )
    print_table(f"Most frequent authors (top {args.top_n})", author_rows(df, args.top_n), ["author", "GND", "docs", "tokens"], fmt)


if __name__ == "__main__":
    main()
