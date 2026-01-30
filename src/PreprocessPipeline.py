import nltk
from nltk import TweetTokenizer
import string
import re
import emoji

class PreprocessPipeline:
  def __init__(self):
    self.tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

    # List of chars to keep
    # point is keep for elipses "..."
    chars_to_keep = "@#?!.'_"
    self.punct_to_remove = "".join([c for c in string.punctuation if c not in chars_to_keep])

  def clean_text(self, text):
    # Converts 😂 to " :face_with_tears_of_joy: "
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Lower
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\b(href|http|https)\b', '', text)

    # Some noise patterns found
    noise_patterns = [
        r'gt',
        r'class[^\w\s]*delicious[^\w\s]*title[^\w\s]*share[^\w\s]*del', # Removes 'gt' (from >)
        r'rel[^\w\s]*nofollow[^\w\s]*target[^\w\s]*blank',              # Specific CSS/HTML string
        r'languagedirection[^\w\s]*ltr',                                 # Specific CSS/HTML string
        r'\b(type|application|atom|xml|feedlinks|href|http|https)\b',     # Directional metadata
    ]

    combined_noise = '|'.join(noise_patterns)
    text = re.sub(combined_noise, '', text)

    # Remove puntuation, keep some special characters
    # We use a translation table here; it's much faster than regex for single characters
    table = str.maketrans('', '', self.punct_to_remove)
    text = text.translate(table)

    text = re.sub(combined_noise, '', text) # re apply

    # Remove extra space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

  def transform(self, text):
    text = self.clean_text(text)
    tokens = self.tweet_tokenizer.tokenize(text)
    return tokens