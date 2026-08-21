# DTA Experiments

This repository hosts tools for various experiments on the Deutsches Textarchiv (DTA).

# Setup

A few dependencies needs to be installed using `uv` with:

```bash
uv sync

# Activate env
source .venv/bin/activate
```

# DTA Dump

A very recent dta dump is used for all experiments and needs to be retrieved first:

```bash
wget https://www.deutschestextarchiv.de/media/download/dta_komplett_2026-02-10_tcf.zip

unzip dta_komplett_2026-02-10.zip
```

The extracted dump is then located under `./dta_komplett_2026-02-10` and serves as our DTA base folder path in the upcoming sections.

# Hugging Face Dataset

## Creation

The script `create_dta_dataset.py` creates parquet files from the DTA dump, so the dataset can easily be shared on the Hugging Face Model Hub.

```bash
python3 create_dta_dataset.py --dta-dir ./dta_komplett_2026-02-10 --output-dir ./hf-dta-documents --num-workers 4
```

It outputs the following:

```bash
5481 documents (union of simple/ and full/)
Wrote hf-dta-documents/train-00000.parquet (1000 rows)
WARNING - skipping ford_pitty_1633: ParseError: mismatched tag: line 16, column 2
Wrote hf-dta-documents/train-00001.parquet (1000 rows)
Wrote hf-dta-documents/train-00002.parquet (1000 rows)
Wrote hf-dta-documents/train-00003.parquet (1000 rows)
Wrote hf-dta-documents/train-00004.parquet (1000 rows)
Wrote hf-dta-documents/train-00005.parquet (480 rows)

Done: 5,480 documents in 6 Parquet shards -> hf-dta-documents/
```

## Schema

The final dataset provides the following columns:

| Column                 | Type           | Description | Example |
|:-----------------------|:---------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
| `id`                   | `string`       | DTA directory name, used as the primary document identifier (`DTADirName`). Falls back to the TCF file name                                                            | `goethe_faust01_1808`                                                                              |
| `dta_id`               | `string`       | Numeric DTA identifier (`DTAID`)                                                                                                                                       | `200006`                                                                                           |
| `urn`                  | `string`       | Persistent URN of the work                                                                                                                                             | `urn:nbn:de:kobv:b4-200006-1`                                                                      |
| `url`                  | `string`       | Landing page of the work at deutschestextarchiv.de                                                                                                                     | `https://www.deutschestextarchiv.de/343016`                                                        |
| `title`                | `string`       | Main title(s) from the TEI title statement. Multiple main titles are joined with `\n`                                                                                  | `Iphigenie auf Tauris`                                                                             |
| `subtitle`             | `string`       | Subtitle(s), joined with `\n`. Null if absent                                                                                                                          | `Ein Schauspiel`                                                                                   |
| `volume`               | `string`       | Volume title (text of `<title type="volume">`). Null if absent                                                                                                         | `Dritter Theil`                                                                                    |
| `part`                 | `string`       | Part title (text of `<title type="part">`). Null if absent                                                                                                             | `Die sentimentalischen Dichter`                                                                    |
| `authors`              | `list<struct>` | Authors as `{surname, forename, gnd}` structs. `gnd` is the GND/PND URI, may be null. Empty list for anonymous works                                                   | `[{"surname": "Reideburg", "forename": "Christoph von", "gnd": "http://d-nb.info/gnd/128862580"}]` |
| `date_published`       | `string`       | Raw publication date string of the original print, as given in the header                                                                                              | `1642`                                                                                             |
| `year`                 | `int64`        | First four-digit number parsed from `date_published`. Null if none found                                                                                               | `1642`                                                                                             |
| `place_of_publication` | `string`       | Place of publication of the original print                                                                                                                             | `Breslau`                                                                                          |
| `publisher`            | `string`       | Publisher name(s), joined with `\n`                                                                                                                                    | `Georgius Baumann`                                                                                 |
| `edition`              | `string`       | Edition statement (text, or the `n` attribute if the element has no text)                                                                                              | `2. Auflage`                                                                                       |
| `num_pages`            | `int64`        | Page count of the original source (`measure type="pages"`)                                                                                                             | `72`                                                                                               |
| `repository`           | `string`       | Holding library of the digitized copy                                                                                                                                  | `Universitätsbibliothek Breslau`                                                                   |
| `shelfmark`            | `string`       | Shelfmark of the copy at the repository                                                                                                                                | `Universitätsbibliothek Breslau, 4 A 277/11 / 343016`                                              |
| `typeface`             | `string`       | Typeface of the original print                                                                                                                                         | `Fraktur`                                                                                          |
| `bibl`                 | `string`       | Short bibliographic citation string                                                                                                                                    | `Reideburg, Christoph von: Kurtze Anleitung: Wie die jetzige böse Zeit/ ... Breslau, 1642.`        |
| `genre_dtamain`        | `string`       | Main genre, DTA scheme                                                                                                                                                 | `Gebrauchsliteratur`                                                                               |
| `genre_dtasub`         | `string`       | Sub genre, DTA scheme                                                                                                                                                  | `Leichenpredigt`                                                                                   |
| `genre_dwds1main`      | `string`       | Main genre, DWDS scheme                                                                                                                                                | `Belletristik`                                                                                     |
| `genre_dwds1sub`       | `string`       | Sub genre, DWDS scheme                                                                                                                                                 | `Prosa`                                                                                            |
| `dta_corpus_labels`    | `list<string>` | DTA (sub)corpus membership labels (`DTACorpus` class codes)                                                                                                            | `["ready", "core"]`                                                                                |
| `language`             | `string`       | ISO 639-3 code of the primary language                                                                                                                                 | `deu`                                                                                              |
| `language_note`        | `string`       | Human-readable language label from the header                                                                                                                          | `(Früh-)Neuhochdeutsch`                                                                            |
| `license`              | `string`       | Per-work license URL exactly as given in the header. Varies across the corpus                                                                                          | `http://creativecommons.org/licenses/by-sa/3.0/de/`                                                |
| `license_family`       | `string`       | Normalized license identifier derived from `license` (http/https, `/de/`, `deed` variants collapsed)                                                                   | `cc-by-sa-3.0`                                                                                     |
| `num_images`           | `int64`        | Number of page images (DTA extent measure)                                                                                                                             | `72`                                                                                               |
| `num_tokens`           | `int64`        | Number of tokens (DTA extent measure)                                                                                                                                  | `11515`                                                                                            |
| `num_types`            | `int64`        | Number of word types (DTA extent measure)                                                                                                                              | `3440`                                                                                             |
| `num_characters`       | `int64`        | Number of characters (DTA extent measure)                                                                                                                              | `78795`                                                                                            |
| `num_sentences`        | `int64`        | Sentence count from the `full/` sentence layer. Null if the work has no `full/` file                                                                                   | `842`                                                                                              |
| `text`                 | `string`       | Historical transcription. Layout-faithful `<text>` layer from `simple/` (line breaks, long s `ſ`, combining diacritics), or space-joined tokens if only `full/` exists | `Kurtze Anleitung:\nWie die jetzige boͤſe Zeit/ darinnen zwar fuͤr ſich ſelbſt/ ...`                 |
| `text_source`          | `string`       | Origin of `text`: `layout` (from `simple/`) or `tokens` (reconstructed from the `full/` token layer)                                                                   | `layout`                                                                                           |
| `text_normalized`      | `string`       | Modern orthography: `full/` token layer with CAB `replace` corrections applied, space-joined. Null if the work has no `full/` file                                     | `Kurze Anleitung : Wie die jetzige böse Zeit / darinnen zwar für sich selbst / ...`                |

### Overview

| statistic                                    | value                 |
|----------------------------------------------|-----------------------|
| Documents                                    | 5,480                 |
| Documents with normalized text (full/ layer) | 5,470 (99.8%)         |
| Documents with at least one author           | 3,104 (56.6%)         |
| Distinct authors (by GND id)                 | 1,317                 |
| Year range                                   | 1472-1987 (1 unknown) |
| Median year                                  | 1836                  |
| Tokens (header measure)                      | 204,582,701           |
| Types (header measure)                       | 33,741,287            |
| Characters (header measure)                  | 1,423,995,435         |
| Characters in `text` column                  | 1,380,752,264         |
| Characters in `text_normalized` column       | 1,370,639,606         |
| Sentences (full/ layer)                      | 10,376,423            |
| Page images                                  | 761,995               |
| Median tokens per document                   | 10,394                |
| Median pages (images) per document           | 16                    |

### License distribution

| license family        | docs      | docs %   | tokens          | tokens %   | license URLs                                                                                                                                                                                                                                                     |
|-----------------------|-----------|----------|-----------------|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cc-by-sa-4.0`        | 2,644     | 48.2%    | 139,433,878     | 68.2%      | http://creativecommons.org/licenses/by-sa/4.0/<br>http://creativecommons.org/licenses/by-sa/4.0/deed.de/<br>https://creativecommons.org/licenses/by-sa/4.0/<br>https://creativecommons.org/licenses/by-sa/4.0/deed.de                                            |
| `cc-by-nc-3.0`        | 1,559     | 28.4%    | 20,418,259      | 10.0%      | http://creativecommons.org/licenses/by-nc/3.0/<br>http://creativecommons.org/licenses/by-nc/3.0/de<br>http://creativecommons.org/licenses/by-nc/3.0/de/                                                                                                          |
| `cc-by-sa-3.0`        | 623       | 11.4%    | 15,287,129      | 7.5%       | http://creativecommons.org/licenses/by-sa/3.0/<br>http://creativecommons.org/licenses/by-sa/3.0/de<br>http://creativecommons.org/licenses/by-sa/3.0/de/<br>https://creativecommons.org/licenses/by-sa/3.0/<br>https://creativecommons.org/licenses/by-sa/3.0/de/ |
| `cc0-1.0`             | 370       | 6.8%     | 16,938,595      | 8.3%       | http://creativecommons.org/publicdomain/zero/1.0/                                                                                                                                                                                                                |
| `cc-by-4.0`           | 215       | 3.9%     | 8,189,539       | 4.0%       | http://creativecommons.org/licenses/by/4.0/<br>http://creativecommons.org/licenses/by/4.0/deed.de<br>https://creativecommons.org/licenses/by/4.0/<br>https://creativecommons.org/licenses/by/4.0/deed.de                                                         |
| `cc-by-sa-2.0`        | 51        | 0.9%     | 3,058,054       | 1.5%       | http://creativecommons.org/licenses/by-sa/2.0/de<br>http://creativecommons.org/licenses/by-sa/2.0/de/                                                                                                                                                            |
| `gutenberg`           | 9         | 0.2%     | 344,316         | 0.2%       | https://web.archive.org/web/20180927123034/http://www.gutenberg.org/wiki/Gutenberg:The_Project_Gutenberg_License                                                                                                                                                 |
| `cc-by-3.0`           | 3         | 0.1%     | 571,346         | 0.3%       | http://creativecommons.org/licenses/by/3.0/de/                                                                                                                                                                                                                   |
| `cc-by-nc-sa-4.0`     | 2         | 0.0%     | 19,672          | 0.0%       | https://creativecommons.org/licenses/by-nc-sa/4.0/<br>https://creativecommons.org/licenses/by-nc-sa/4.0/deed.de                                                                                                                                                  |
| `mdz-copyright`       | 1         | 0.0%     | 24,964          | 0.0%       | http://mdz.bib-bvb.de/copyright.htm                                                                                                                                                                                                                              |
| `noc-nc-1.0`          | 1         | 0.0%     | 125,859         | 0.1%       | http://rightsstatements.org/vocab/NoC-NC/1.0/                                                                                                                                                                                                                    |
| `nug-kkn`             | 1         | 0.0%     | 55,702          | 0.0%       | http://www.deutsche-digitale-bibliothek.de/lizenzen/nug-kkn/                                                                                                                                                                                                     |
| `out-of-copyright-nc` | 1         | 0.0%     | 115,388         | 0.1%       | https://www.europeana.eu/portal/de/rights/out-of-copyright-non-commercial.html                                                                                                                                                                                   |
| **total**             | **5,480** | **100%** | **204,582,701** | **100%**   |                                                                                                                                                                                                                                                                  |

### Temporal distribution (50-year bins)

| period    | docs      | sentences      | tokens          | characters        | token share   |
|-----------|-----------|----------------|-----------------|-------------------|---------------|
| 1450–1499 | 15        | 1,899          | 177,258         | 997,389           | 0.1%          |
| 1500–1549 | 17        | 10,469         | 308,471         | 1,915,965         | 0.2%          |
| 1550–1599 | 98        | 139,035        | 3,572,336       | 23,618,803        | 1.7%          |
| 1600–1649 | 422       | 433,762        | 9,757,100       | 66,033,889        | 4.8%          |
| 1650–1699 | 691       | 1,118,361      | 23,338,594      | 159,938,400       | 11.4%         |
| 1700–1749 | 435       | 1,143,293      | 24,779,141      | 171,116,209       | 12.1%         |
| 1750–1799 | 583       | 1,566,582      | 30,933,941      | 214,031,382       | 15.1%         |
| 1800–1849 | 1,618     | 2,000,919      | 40,798,688      | 284,536,219       | 19.9%         |
| 1850–1899 | 1,143     | 3,086,868      | 55,792,116      | 393,248,560       | 27.3%         |
| 1900–1949 | 444       | 737,908        | 13,114,628      | 95,203,044        | 6.4%          |
| 1950–1999 | 13        | 137,150        | 2,005,488       | 13,321,270        | 1.0%          |
| unknown   | 1         | 177            | 4,940           | 34,305            | 0.0%          |
| **total** | **5,480** | **10,376,423** | **204,582,701** | **1,423,995,435** | **100%**      |

### Text source

| text_source   | docs   | docs %   | tokens      | tokens %   |
|---------------|--------|----------|-------------|------------|
| `layout`      | 5,105  | 93.2%    | 185,164,389 | 90.5%      |
| `tokens`      | 375    | 6.8%     | 19,418,312  | 9.5%       |

### Genre (DTA main)

| genre_dtamain                                                                       | docs   | docs %   | tokens     | tokens %   |
|-------------------------------------------------------------------------------------|--------|----------|------------|------------|
| unknown                                                                             | 3,468  | 63.3%    | 69,372,311 | 33.9%      |
| Fachtext                                                                            | 743    | 13.6%    | 78,106,238 | 38.2%      |
| Belletristik                                                                        | 561    | 10.2%    | 33,626,235 | 16.4%      |
| Gebrauchsliteratur                                                                  | 536    | 9.8%     | 22,981,508 | 11.2%      |
| Wissenschaftliche Abhandlungen in Form gedruckter Briefe                            | 50     | 0.9%     | 89,564     | 0.0%       |
| Abhandlungen in Zeitschriften, Sammelbänden etc.                                    | 36     | 0.7%     | 145,342    | 0.1%       |
| Ankündigungen, Berichtigungen und kurze Nachrichten                                 | 29     | 0.5%     | 25,687     | 0.0%       |
| Berliner Akademiereden/-schriften und andere Reden                                  | 17     | 0.3%     | 71,027     | 0.0%       |
| Rezensionen                                                                         | 7      | 0.1%     | 8,514      | 0.0%       |
| Pariser Akademiereden/-schriften                                                    | 6      | 0.1%     | 16,728     | 0.0%       |
| Vorworte und andere Beiträge Humboldts in Schriften anderer Autoren, Lexikonartikel | 6      | 0.1%     | 20,581     | 0.0%       |
| Gelegenheitsschrift                                                                 | 5      | 0.1%     | 49,906     | 0.0%       |
| Journalismus                                                                        | 5      | 0.1%     | 54,829     | 0.0%       |
| Albumblätter                                                                        | 3      | 0.1%     | 812        | 0.0%       |
| Gutachten                                                                           | 3      | 0.1%     | 6,339      | 0.0%       |
| Sonderdrucke und andere Grenzfälle zu selbständig erschienenen Schriften            | 2      | 0.0%     | 3,358      | 0.0%       |
| Wissenschaft                                                                        | 2      | 0.0%     | 3,124      | 0.0%       |
| Sachliteratur                                                                       | 1      | 0.0%     | 598        | 0.0%       |

### Genre (DTA sub, top 10)

| genre_dtasub        | docs   | docs %   | tokens     | tokens %   |
|---------------------|--------|----------|------------|------------|
| unknown             | 3,628  | 66.2%    | 69,896,619 | 34.2%      |
| Leichenpredigt      | 334    | 6.1%     | 3,495,630  | 1.7%       |
| Roman               | 163    | 3.0%     | 11,534,905 | 5.6%       |
| Prosa               | 128    | 2.3%     | 10,291,868 | 5.0%       |
| Lyrik               | 114    | 2.1%     | 4,731,157  | 2.3%       |
| Drama               | 82     | 1.5%     | 2,330,599  | 1.1%       |
| Recht               | 79     | 1.4%     | 9,743,726  | 4.8%       |
| Philosophie         | 76     | 1.4%     | 5,426,303  | 2.7%       |
| Historiographie     | 46     | 0.8%     | 7,221,703  | 3.5%       |
| Medizin             | 46     | 0.8%     | 6,013,804  | 2.9%       |
| *other (86 values)* | 784    | 14.3%    | 73,896,387 | 36.1%      |

### Genre (DWDS main)

| genre_dwds1main    | docs   | docs %   | tokens     | tokens %   |
|--------------------|--------|----------|------------|------------|
| Zeitung            | 2,036  | 37.2%    | 17,152,284 | 8.4%       |
| Gebrauchsliteratur | 1,718  | 31.4%    | 58,283,320 | 28.5%      |
| Wissenschaft       | 953    | 17.4%    | 87,148,247 | 42.6%      |
| Belletristik       | 773    | 14.1%    | 41,998,850 | 20.5%      |

### Genre (DWDS sub, top 10)

| genre_dwds1sub          | docs   | docs %   | tokens      | tokens %   |
|-------------------------|--------|----------|-------------|------------|
| unknown                 | 2,036  | 37.2%    | 17,152,284  | 8.4%       |
| Leichenpredigt          | 336    | 6.1%     | 3,517,225   | 1.7%       |
| Zeitschrift             | 217    | 4.0%     | 1,697,733   | 0.8%       |
| Theologie               | 214    | 3.9%     | 10,486,746  | 5.1%       |
| Brief                   | 204    | 3.7%     | 84,325      | 0.0%       |
| Roman                   | 198    | 3.6%     | 13,211,036  | 6.5%       |
| Gesellschaft            | 165    | 3.0%     | 3,852,413   | 1.9%       |
| Novelle                 | 126    | 2.3%     | 3,031,211   | 1.5%       |
| Lyrik                   | 125    | 2.3%     | 5,090,960   | 2.5%       |
| Gelegenheitsschrift:Tod | 116    | 2.1%     | 103,280     | 0.1%       |
| *other (123 values)*    | 1,743  | 31.8%    | 146,355,488 | 71.5%      |

### Typeface

| typeface                  | docs   | docs %   | tokens      | tokens %   |
|---------------------------|--------|----------|-------------|------------|
| Fraktur                   | 4,478  | 81.7%    | 164,623,059 | 80.5%      |
| Antiqua                   | 427    | 7.8%     | 35,856,371  | 17.5%      |
| Schwabacher               | 338    | 6.2%     | 724,934     | 0.4%       |
| Handschrift               | 218    | 4.0%     | 1,315,690   | 0.6%       |
| Jean-Paul-Fraktur         | 9      | 0.2%     | 1,409,611   | 0.7%       |
| Current                   | 2      | 0.0%     | 1,963       | 0.0%       |
| Rotunda                   | 2      | 0.0%     | 313,225     | 0.2%       |
| Antiqua (Schreibmaschine) | 1      | 0.0%     | 34,176      | 0.0%       |
| Antiqua, Kursive          | 1      | 0.0%     | 18,883      | 0.0%       |
| Antqiua                   | 1      | 0.0%     | 3,527       | 0.0%       |
| Frakur                    | 1      | 0.0%     | 10,103      | 0.0%       |
| Textur                    | 1      | 0.0%     | 270,599     | 0.1%       |
| unknown                   | 1      | 0.0%     | 560         | 0.0%       |

### Language

| language   | docs   | docs %   | tokens      | tokens %   |
|------------|--------|----------|-------------|------------|
| `deu`      | 5,477  | 99.9%    | 204,523,746 | 100.0%     |
| `lat`      | 2      | 0.0%     | 56,714      | 0.0%       |
| `gml`      | 1      | 0.0%     | 2,241       | 0.0%       |

### DTA corpus labels (a document can carry several)

| label                  | docs   | docs %   | tokens      | tokens %   |
|------------------------|--------|----------|-------------|------------|
| `ready`                | 5,480  | 100.0%   | 204,582,701 | 100.0%     |
| `core`                 | 1,478  | 27.0%    | 130,114,948 | 63.6%      |
| `china`                | 1,149  | 21.0%    | 111,416,242 | 54.5%      |
| `mkhz1`                | 625    | 11.4%    | 3,996,672   | 2.0%       |
| `nrhz`                 | 530    | 9.7%     | 4,290,483   | 2.1%       |
| `aedit`                | 335    | 6.1%     | 3,508,098   | 1.7%       |
| `mts`                  | 320    | 5.8%     | 18,006,989  | 8.8%       |
| `lefevre`              | 319    | 5.8%     | 476,390     | 0.2%       |
| `dtae`                 | 244    | 4.5%     | 21,475,420  | 10.5%      |
| `mkhz2`                | 222    | 4.1%     | 2,169,371   | 1.1%       |
| `correspondent`        | 204    | 3.7%     | 925,755     | 0.5%       |
| `tevo`                 | 202    | 3.7%     | 5,682,395   | 2.8%       |
| `sanders-briefe`       | 190    | 3.5%     | 61,890      | 0.0%       |
| `hab`                  | 184    | 3.4%     | 10,256,959  | 5.0%       |
| `augsburgerallgemeine` | 181    | 3.3%     | 2,761,778   | 1.3%       |
| `avh`                  | 172    | 3.1%     | 443,668     | 0.2%       |
| `wikisource`           | 169    | 3.1%     | 5,787,543   | 2.8%       |
| `sbb_funeralschriften` | 112    | 2.0%     | 77,981      | 0.0%       |
| `tdef`                 | 109    | 2.0%     | 940,772     | 0.5%       |
| `novellenschatz`       | 87     | 1.6%     | 1,643,336   | 0.8%       |
| `blumenbach`           | 33     | 0.6%     | 2,688,412   | 1.3%       |
| `epoetics`             | 23     | 0.4%     | 3,074,730   | 1.5%       |
| `frauenstudium`        | 22     | 0.4%     | 134,198     | 0.1%       |
| `psyleko`              | 19     | 0.3%     | 2,075,500   | 1.0%       |
| `ntsm`                 | 11     | 0.2%     | 32,139      | 0.0%       |
| `avhkv`                | 9      | 0.2%     | 636,539     | 0.3%       |
| `briefjeanpaul`        | 9      | 0.2%     | 1,409,611   | 0.7%       |
| `gutenberg_org`        | 8      | 0.1%     | 325,940     | 0.2%       |
| `gutzkow`              | 7      | 0.1%     | 478,246     | 0.2%       |
| `dwds1`                | 5      | 0.1%     | 465,688     | 0.2%       |
| `gei`                  | 2      | 0.0%     | 57,474      | 0.0%       |
| `gutenberg_de`         | 2      | 0.0%     | 116,184     | 0.1%       |
| `gwb`                  | 2      | 0.0%     | 80,666      | 0.0%       |
| `urmel`                | 2      | 0.0%     | 1,068       | 0.0%       |
| `zbk`                  | 2      | 0.0%     | 48,127      | 0.0%       |
| `greflinger`           | 1      | 0.0%     | 9,079       | 0.0%       |
| `grenzboten`           | 1      | 0.0%     | 116,208     | 0.1%       |

### Place of publication (top 10)

| place                | docs   | docs %   | tokens      | tokens %   |
|----------------------|--------|----------|-------------|------------|
| Berlin               | 604    | 11.0%    | 30,616,946  | 15.0%      |
| Köln                 | 532    | 9.7%     | 4,682,997   | 2.3%       |
| Leipzig              | 425    | 7.8%     | 41,950,384  | 20.5%      |
| Hamburg              | 330    | 6.0%     | 5,514,413   | 2.7%       |
| Augsburg             | 256    | 4.7%     | 4,581,351   | 2.2%       |
| Altstrelitz          | 172    | 3.1%     | 57,702      | 0.0%       |
| unknown              | 160    | 2.9%     | 980,933     | 0.5%       |
| München              | 153    | 2.8%     | 3,434,662   | 1.7%       |
| Frankfurt (Main)     | 141    | 2.6%     | 11,430,873  | 5.6%       |
| Königsberg           | 133    | 2.4%     | 926,843     | 0.5%       |
| *other (246 values)* | 2,574  | 47.0%    | 100,405,597 | 49.1%      |

### Most frequent authors (top 10)

| author                       | GND                            | docs   | tokens    |
|------------------------------|--------------------------------|--------|-----------|
| Humboldt, Alexander von      | http://d-nb.info/gnd/118554700 | 183    | 1,504,828 |
| Sanders, Daniel              | http://d-nb.info/gnd/119242044 | 173    | 69,412    |
| Dach, Simon                  | http://d-nb.info/gnd/11852321X | 112    | 77,981    |
| N. N.                        |                                | 51     | 1,882,041 |
| Blumenbach, Johann Friedrich | http://d-nb.info/gnd/116208503 | 35     | 2,814,553 |
| Sattler, Basilius            | http://d-nb.info/gnd/116974788 | 26     | 352,365   |
| Goethe, Johann Wolfgang von  | http://d-nb.info/gnd/118540238 | 24     | 1,064,748 |
| Herder, Johann Gottfried von | http://d-nb.info/gnd/118549553 | 23     | 690,276   |
| Grimm, Jacob                 | http://d-nb.info/gnd/118542257 | 17     | 2,441,309 |
| Fontane, Theodor             | http://d-nb.info/gnd/118534262 | 16     | 1,262,140 |

## Upload

The following was used to upload the parquet files to the Hugging Face Model Hub:

```bash
hf upload "histde/dta-documents" ./hf-dta-documents \
  --repo-type dataset \
  --private \
  --include "*.parquet" \
  --commit-message "feat: add initial dataset"
```

# Character-based xLSTM

For further experiments on OCR quality scoring, we first train a character-based xLSTM model on the DTA documents.

## Tokenizer

The first step is to build a character tokenizer on the DTA documents. The `build_tokenizer.py` script uses the `text` column, normalizes it with the rules defined in `text_normalization.py` (such as de-hyphenation, NFC, lowercasing, `ſ` -> `s`, `uͤ` -> `ü`, quote/dash unification, digits -> `0`, whitespace collapsing), counts characters and keeps the most frequent ones up to `--vocab-size` (256 by default, so data shards fit in `uint8`), plus `<eos>` (id 0) and `<unk>` (id 1).

The following command can be used for that:

```bash
python build_tokenizer.py \
  --dataset "histde/dta-documents" \
  --split train \
  --output-dir dta_xlstm/tokenizer \
  --vocab-size 256 \
  --num-workers 8
```

This outputs:

```bash
Counting characters in histde/dta-documents (split train) ...
  500 rows streamed, 47,712,027 characters
  1000 rows streamed, 270,289,702 characters
  1500 rows streamed, 480,631,065 characters
  2000 rows streamed, 565,806,549 characters
  2500 rows streamed, 779,292,258 characters
  3000 rows streamed, 819,022,815 characters
  3500 rows streamed, 859,108,062 characters
  4000 rows streamed, 887,371,455 characters
  4500 rows streamed, 920,697,707 characters
  5000 rows streamed, 1,090,396,475 characters
5480 usable documents, 1,366,003,516 characters, 1276 distinct
Vocabulary: 256 entries, <unk> rate 0.00398%
Done -> dta_xlstm/tokenizer/ (tokenizer.json, tokenizer_config.json, vocab.json, tokenizer_stats.json)
```

The result is a Hugging Face Fast Tokenizer (`tokenizer.json` + `tokenizer_config.json`). Another written files are the `vocab.json`, which is a human-readable vocab and `tokenizer_stats.json`, that includes some tokenizer stats.

The HF-compatible tokenizer can be loaded and tested with:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dta_xlstm/tokenizer")
tokenizer.tokenize("Wie die jetzige boͤſe Zeit/ 1642")
# ['w', 'i', 'e', ' ', 'd', 'i', 'e', ' ', 'j', 'e', 't', 'z', 'i', 'g', 'e', ' ', 'b', 'ö', 's', 'e', ' ', 'z', 'e', 'i', 't', '/', ' ', '0', '0', '0', '0']
```

## Data Preprocessing

The second step encodes the DTA documents with that tokenizer into Numpy-compatible shards that the pretraining script is later reading-in. Documents are streamed, encoded and written as they arrive:

```bash
python prepare_xlstm_data.py \
  --tokenizer dta_xlstm/tokenizer \
  --dataset "histde/dta-documents" \
  --split train \
  --output-dir dta_xlstm/data \
  --save-every-n-chars 100000000 \
  --num-workers 8
```

This outputs:

```bash
Tokenizer: dta_xlstm/tokenizer (256 tokens, uint8 shards)
Encoding histde/dta-documents (split train) ...
  500 rows streamed, 500 usable, 47,712,027 characters
Writing numpy shard: dta_xlstm/data/dta_000000.npy (103,041,651 characters)
Writing numpy shard: dta_xlstm/data/dta_000001.npy (100,520,151 characters)
  1000 rows streamed, 1000 usable, 270,289,702 characters
Writing numpy shard: dta_xlstm/data/dta_000002.npy (100,100,389 characters)
Writing numpy shard: dta_xlstm/data/dta_000003.npy (100,068,658 characters)
  1500 rows streamed, 1500 usable, 480,631,065 characters
Writing numpy shard: dta_xlstm/data/dta_000004.npy (100,142,372 characters)
  2000 rows streamed, 2000 usable, 565,806,549 characters
Writing numpy shard: dta_xlstm/data/dta_000005.npy (100,196,485 characters)
Writing numpy shard: dta_xlstm/data/dta_000006.npy (100,055,973 characters)
  2500 rows streamed, 2500 usable, 779,292,258 characters
Writing numpy shard: dta_xlstm/data/dta_000007.npy (100,147,565 characters)
  3000 rows streamed, 3000 usable, 819,022,815 characters
  3500 rows streamed, 3500 usable, 859,108,062 characters
  4000 rows streamed, 4000 usable, 887,371,455 characters
Writing numpy shard: dta_xlstm/data/dta_000008.npy (100,031,857 characters)
  4500 rows streamed, 4500 usable, 920,697,707 characters
Writing numpy shard: dta_xlstm/data/dta_000009.npy (101,487,908 characters)
  5000 rows streamed, 5000 usable, 1,090,396,475 characters
Writing numpy shard: dta_xlstm/data/dta_000010.npy (100,227,277 characters)
Writing numpy shard: dta_xlstm/data/dta_000011.npy (100,305,673 characters)
Writing numpy shard: dta_xlstm/data/dta_000012.npy (101,998,777 characters)
Writing numpy shard: dta_xlstm/data/dta_000013.npy (57,684,260 characters)
5480 usable documents, 1,366,003,516 characters, <unk> rate 0.00398%
Done -> dta_xlstm/data/ (14 shards, tokenizer, metadata.json)
```

## Training

The character-based xLSTM model can be trained with:

```bash
python train_xlstm.py \
  --data-dir dta_xlstm/data \
  --output_dir dta_xlstm/model \
  --run_name dta-xlstm-20m \
  --hidden_size 384 \
  --num_blocks 11 \
  --num_heads 3 \
  --block_size 2048 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 2 \
  --token_budget 2500000000 \
  --learning_rate 3e-3 \
  --min_lr_rate 0.1 \
  --adam_beta1 0.99 \
  --adam_beta2 0.95 \
  --weight_decay 0.1 \
  --max_grad_norm 0.5 \
  --bf16 \
  --chunkwise_kernel chunkwise--triton_xl_chunk \
  --dataloader_num_workers 2 \
  --logging_steps 100 \
  --save_steps 2500 \
  --save_total_limit 3 \
  --seed 42
```

# 📝 Changelog

* 21.08.2026: Initial release of this repo.

# 🤖 AI disclosure

The repository introduces some "AI disclosure" rules:

* The README's are written 100% by a human. Aren't we all tired of Em-Dashes, and Oxford commata?
* The Python scripts always include an AI disclosure block, that is stating: a) which model was used, b) the AI-generated level and c) the level of human review.

Here's an example:

```python
"""<One-line description of what the file does.>

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   mostly        # fully | mostly | partially | none
    Human-Reviewed: partially     # fully | partially | minimally | none
"""
```

The rules are maintained as a standalone ruleset in my [AI Disclosure](https://github.com/stefan-it/ai-disclosure) repo.

# ⚖️ License

The repo and its content is licensed under [Apache License 2.0](LICENSE).

The AI-generated content is mainly produced by Claude Code, using a Pro or Max plan. Here's a super interesting [blog post](https://www.oreilly.com/radar/who-owns-the-code-claude-wrote/) that mentions the difference between API usage and subscription plans in the copyright world.
