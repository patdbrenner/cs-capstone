# IT Support Ticket Classifier

An automated IT support ticket classification system built for the WGU C964 Computer Science Capstone.
The application uses a Random Forest classifier trained on TF-IDF text features to automatically route IT support
tickets into one of eight categories, removing the need for manual triage.

## Categories

Hardware, HR Support, Access, Storage, Purchase, Internal Project, Administrative rights, Miscellaneous

## Project Structure

```
it-ticket-classifier/
├── app/                    # Streamlit application
│   ├── __init__.py
│   ├── auth.py             # Password gate
│   ├── classify.py         # Ticket submission form and classification
│   ├── dashboard.py        # Model info, recent predictions, and three visualizations
│   ├── data.py             # Cached loading of the model, vectorizer, test split, and metadata
│   └── prediction_log.py   # Appends and reads logged predictions
├── app.py                  # Streamlit entry point
├── data/                   # Raw dataset (Kaggle IT Service Management Ticket Classification dataset)
├── models/                 # Trained model, vectorizer, and metadata (model.pkl tracked with Git LFS)
├── outputs/                # Generated charts and prediction log
├── src/                    # Core ML pipeline
│   ├── __init__.py
│   ├── eval.py             # Model evaluation: weighted F1, confusion matrix, feature importance
│   ├── explore.py          # Data exploration: class distribution, missing values, text length
│   ├── preprocess.py       # Text cleaning pipeline (lowercase, punctuation, stop words, stemming, tokenize)
│   ├── train.py            # Model training with RandomizedSearchSV hyperparameter tuning
│   └── visualizations.py   # Shared chart-build functions used by eval.py and the dashboard
└── tests/                  # Unit tests
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd it-ticket-classifier
```

This repo uses Git LFS for `models/model.pkl`. Install Git LFS before cloning:

```bash
git lfs install
```

### 2. Install dependencies

```bash
pipenv install --dev
```

NLTK resources (`punkt`, `punkt_tab`, `stopwords`) are included in the repo.

### 3. Configure secrets

Copy the example secrets file and set your own password:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and replace the example password. File is gitignored.

## Usage

Run each step from the project root.

**Explore the dataset:**
```bash
pipenv run python -m src.explore
```

**Train the model:**
```bash
pipenv run python -m src.train
```
This saves `model.pkl`, `vectorizer.pkl`, `metadata.json` and `test_split.pkl` to `models/`.

**Evaluate the model:**
```bash
pipenv run python -m src.eval
```
This saves the confusion matrix and feature importance charts to `outputs/`.

**Run the web app:**
```bash
pipenv run streamlit run app.py
```

**Run tests:**
```bash
pipenv run pytest tests/ -v
```

## Model Performance

The trained model achieves a weighted F1-score above the 75% target on the test set. Metrics are saved to `models/metadata.json`
after each training and displayed on the app's Model Dashboard tab, along with the confusion matrix and feature importance
chart in `outputs/`.

## Deployment

This app is deployed to Streamlit Community Cloud, which deploys from GitHub. As the primary repository is hosted on GitLab,
the repository is mirrored to GitHub for deployment purposes.