"""Shared text normalization for the DTA character language model.

Maps historical German print (DTA transcriptions, 1600-1900) and DDB newspaper
OCR output into one common character space, so that training corpus and scored
pages differ by OCR noise only - not by transcription conventions:

  * dehyphenation of line breaks ("-\\n" and "⸗\\n");
  * Unicode NFC, lowercasing;
  * historical glyphs: long s (ſ) -> s, r rotunda (ꝛ) -> r,
    combining small e (aͤ/oͤ/uͤ) -> umlauts, ⸗ -> "-";
  * typographic variants: all double/single quote styles -> "/', dashes -> "-";
  * all digits -> "0" (numbers carry no OCR quality signal, this collapses
    dates/prices/page numbers into one pattern);
  * whitespace runs -> single space.

`to_char_tokens` renders normalized text as space-separated characters (word
boundary = "▁") - the input format of the KenLM character n-gram model.

AI Disclosure:
    Models:         Claude Fable 5 (claude-fable-5)
    AI-Generated:   fully         # fully | mostly | partially | none
    Human-Reviewed: partially     # fully | partially | minimally | none
"""

import re
import unicodedata

WORD_BOUNDARY = "▁"

_COMBINING_E = {"aͤ": "ä", "oͤ": "ö", "uͤ": "ü"}

_CHAR_MAP = str.maketrans({
    "ſ": "s",   # long s
    "ꝛ": "r",   # r rotunda
    "⸗": "-",   # double oblique hyphen
    "„": '"', "“": '"', "”": '"', "»": '"', "«": '"',
    "‘": "'", "’": "'", "‚": "'",
    "–": "-", "—": "-",
})

_DIGITS = re.compile(r"\d")
_WHITESPACE = re.compile(r"\s+")


def normalize(text):
    text = text.replace("⸗\n", "").replace("-\n", "")
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    for sequence, replacement in _COMBINING_E.items():
        text = text.replace(sequence, replacement)
    text = text.replace("ͤ", "")
    text = text.translate(_CHAR_MAP)
    text = _DIGITS.sub("0", text)
    return _WHITESPACE.sub(" ", text).strip()


def to_char_tokens(text):
    return " ".join(WORD_BOUNDARY if char == " " else char for char in text)
