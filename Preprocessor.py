

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    """Lowercase + strip out links, numbers, and punctuation."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                # remove emails
    text = re.sub(r"\d+", " ", text)                    # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()             # collapse spaces
    return text


class TextPreprocessor:
    """
    Wraps TfidfVectorizer so training and prediction always clean
    text the exact same way.
    """

    def __init__(self, max_features=3000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",   # drop common filler words automatically
        )

    def fit_transform(self, texts):
        """Used ONLY during training: learns the vocabulary AND
        converts the texts to TF-IDF vectors."""
        cleaned = [clean_text(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned)

    def transform(self, texts):
        """Used during prediction: reuses the vocabulary learned
        during training to convert new text to TF-IDF vectors."""
        cleaned = [clean_text(t) for t in texts]
        return self.vectorizer.transform(cleaned)