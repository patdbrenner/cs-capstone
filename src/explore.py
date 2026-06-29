from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualizations import build_distribution_figure

logger = logging.getLogger(__name__)

DATA_PATH = 'data/all_tickets_processed_improved_v3.csv'
TEXT_COLUMN = 'Document'
LABEL_COLUMN = 'Topic_group'

OUTPUT_DIR = Path('outputs')
DISTRIBUTION_CHART_PATH = OUTPUT_DIR / 'class_distribution.png'


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    '''
    Load the raw dataset from CSV.
    
    Args:
        path: Path to the CSV file relative to project root.
        
    Returns:
        DataFrame containing the raw ticket data.
    '''
    logger.info('Loading dataset from %s', path)
    df = pd.read_csv(path)
    logger.info('Loaded %d rows and %d columns', df.shape[0], df.shape[1])
    return df

def report_missing_values(df: pd.DataFrame) -> pd.Series:
    '''
    Report count of missing values per column.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Series of missing value counts indexed by column name.
    '''
    missing = df.isnull().sum()
    logger.info('Missing values per column:\n%s', missing)
    return missing

def report_class_distribution(df: pd.DataFrame, label_column: str = LABEL_COLUMN) -> pd.Series:
    '''
    Report the count and percentage of each class label.
    
    Args:
        df: Input DataFrame.
        label_column: Name of the column containing class labels.
        
    Returns:
        Series of class counts indexed by label.
    '''
    counts = df[label_column].value_counts()
    percentages = df[label_column].value_counts(normalize=True) * 100
    logger.info(f'Class distribution:\n{pd.DataFrame({'count': counts, 'percent': percentages.round(2)})}')

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig = build_distribution_figure(counts)
    fig.savefig(DISTRIBUTION_CHART_PATH)
    plt.close(fig)
    logger.info(f'Saved class distribution chart to {DISTRIBUTION_CHART_PATH}')

    return counts

def report_text_length_stats(df: pd.DataFrame, text_column: str = TEXT_COLUMN) -> pd.Series:
    '''
    Report summary statistics on text length for the text column.
    
    Args:
        df: Input DataFrame.
        text_column: Name of column containing raw text.
        
    Returns:
        Series of summary statistics (count, mean, std, min, max, etc.).
    '''
    lengths = df[text_column].astype(str).str.len()
    stats = lengths.describe()
    logger.info('Text length statistics (chars):\n%s', stats)
    return stats

def run_exploration(path: str = DATA_PATH) -> pd.DataFrame:
    '''
    Run the full data exploration report and return the loaded DataFrame.
    
    Args:
        path: Path to the CSV file relative to project root.
        
    Returns:
        The loaded DataFrame, in case needed further.
    '''
    df = load_data(path)
    report_missing_values(df)
    report_class_distribution(df)
    report_text_length_stats(df)
    return df

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    run_exploration()