import glob
import pandas as pd
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from pathlib import Path

# Hardcoded constants (only file pattern from config)
TEXT_COLUMN = "clean_text"
LABEL_COLUMN = "category"
TOKENS_COLUMN = "tokens"
LANGUAGE = "english"

# Fixed patterns
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")

class SimplePreprocessor:
    def __init__(self, config_filename: str = "config/config.yaml"):
        config_path = Path(__file__).parent.parent.parent / config_filename
        """Load ONLY file pattern from config"""
        self.config = self._load_config(config_path)
        self._init_nltk()
    
    def _load_config(self, config_path: str):
        with open(Path(config_path)) as f:
            return yaml.safe_load(f)
    
    def _init_nltk(self):
        """Download NLTK once"""
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        self.stop_words = set(stopwords.words(LANGUAGE))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text: str):
        """Fixed preprocessing pipeline"""
        # Normalize
        text = str(text).lower()
        text = URL_PATTERN.sub(" ", text)
        HTML_PATTERN.sub(" ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter stopwords + short words
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        return tokens
    
    def load_and_preprocess(self) -> pd.DataFrame:
        """Load + preprocess using config file pattern"""
        csv_files = glob.glob(self.config['preprocessing']['input_files'])
        if not csv_files:
            raise FileNotFoundError(f"No files: {self.config['preprocessing']['input_files']}")
        
        print(f"📂 Loading {len(csv_files)} files...")
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        
        print(f"🔄 Preprocessing '{TEXT_COLUMN}'...")
        df[TOKENS_COLUMN] = df[TEXT_COLUMN].astype(str).apply(self.preprocess_text)
        
        # Clean NaNs (hardcoded logic)
        print("🧹 Cleaning NaNs...")
        df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
        
        print(f"✅ Shape: {df.shape}")
        print(f"Labels: {df[LABEL_COLUMN].value_counts().to_dict()}")
        return df

# Usage
if __name__ == "__main__":
    preprocessor = SimplePreprocessor()
    df = preprocessor.load_and_preprocess()
    print(df[[TEXT_COLUMN, TOKENS_COLUMN, LABEL_COLUMN]].head())
