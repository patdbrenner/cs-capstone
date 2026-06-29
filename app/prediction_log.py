from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LOG_DIR = Path('outputs')
PREDICTION_LOG_PATH = LOG_DIR / 'predictions.jsonl'

def log_prediction(title: str, description: str, predicted_category: str, confidence: float) -> None:
    '''
    Append a single prediction record to the prediction log.
    
    Args:
        title: Ticket title submitted by the user.
        description: Ticket body submitted by the user.
        predicted_category: Category predicted by the model.
        confidence: Predicted probability for the predicted category.
    '''
    LOG_DIR.mkdir(exist_ok=True)
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'title': title,
        'description': description,
        'predicted_category': predicted_category,
        'confidence': confidence,}
    with open(PREDICTION_LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')
    logger.info(f'Logged prediction: {predicted_category} ({confidence:.2%})')

def load_prediction_log() -> pd.DataFrame:
    '''
    Load all logged predictions as a DataFrame, LiFo.
    
    Returns:
        DataFrame of prediction records, empty if no log is present.
    '''
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame(
            columns=['timestamp', 'title', 'description', 'predicted_category', 'confidence'])
    
    records = []
    with open(PREDICTION_LOG_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    logger.info(f'Loaded {len(df)} prediction log records')
    return df