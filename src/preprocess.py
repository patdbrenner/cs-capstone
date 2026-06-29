from __future__ import annotations

import logging
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

NLTK_DATA_DIR = Path(__file__).resolve().parent.parent / 'nltk_data'

if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

# def ensure_nltk_resources() -> None:
#     '''Download NLTK resources if not already present.'''
#     resources = {
#         'tokenizers/punkt': 'punkt',
#         'tokenizers/punkt_tab': 'punkt_tab',
#         'corpora/stopwords': 'stopwords',}
    
#     for path, package in resources.items():
#         try:
#             nltk.data.find(path)
#             logger.debug(f'NLTK resource already present: {package}')
#         except LookupError:
#             logger.info(f'NLTK resource "{package}" not found, downloading')
#             success = nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=False)
#             if not success:
#                 logger.error(f'Failed to download NLTK resource: {package}')
#                 raise RuntimeError(
#                     f'Could not download required NLTK resource "{package}". '
#                     'Check network access and try running '
#                     f'"python -m nltk.downloader {package}" manually.')

# ensure_nltk_resources

_STEMMER = PorterStemmer()
_STOP_WORDS: set[str] = set(stopwords.words('english'))
_PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    
    text = text.lower()
    text = text.translate(_PUNCTUATION_TABLE)
    
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP_WORDS and t.strip()]
    stemmed = [_STEMMER.stem(t) for t in tokens]

    return ' '.join(stemmed)

def clean_series(text_series: pd.Series) -> pd.Series:
    logger.info(f'Cleaning {len(text_series)} text records')
    cleaned = text_series.apply(clean_text)
    logger.info('Finished cleaning text records')
    return cleaned