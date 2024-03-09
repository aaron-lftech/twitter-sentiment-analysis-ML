import re
from collections import Counter

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Dict with emojis and their meanings
emojis = {
    ';-)': 'wink', ':P': 'raspberry', ':O': 'surprised', '=^.^=': 'cat',
    ':-D': 'smile', 'O.o': 'confused', '(:-D': 'gossip', '<(-_-)>': 'robot',
    ':)': 'smile', ':-(': 'sad', ';D': 'wink', 'O*-)': 'angel',
    ':-)': 'smile', ':(': 'sad', ':-E': 'vampire', ':@': 'shocked',
    ':-@': 'shocked', ':\\': 'annoyed', ':-&': 'confused', '$_$': 'greedy',
    ':-!': 'confused', ':-0': 'yell', ":'-)": 'tears of joy', ';)': 'wink',
    'd[-_-]b': 'dj', 'O:-)': 'angel', ':-#': 'mute', ':X': 'mute',
    ':^)': 'smile', '@@': 'eyeroll', ':#': 'mute', ':-$': 'embarrassed',
    ':]': 'smile', '8-)': 'cool', ':-]': 'smile', ':-|': 'neutral',
    '(^_^)': 'joy', '(>_<)': 'frustration', '(T_T)': 'cry', '(^-^*)': 'shy',
    '(ಠ_ಠ)': 'disapproval', '(*_*)': 'amazed', '(づ｡◕‿‿◕｡)づ': 'hug',
    ':*': 'kiss', '¯\\_(ツ)_/¯': 'shrug', '>:[': 'grumpy'
}
# Convert all to lowercase
emojis = {key.lower(): value for key, value in emojis.items()}

# List of stop words in English
nltk.download('stopwords')
stopword_list = stopwords.words('english')

# Download wordnet for lemmatizer
nltk.download('wordnet')

def preprocess_text(text_data,threshold=0):
    processed_text = []
    all_words = []

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Define regex patterns
    url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    user_pattern = '@[^\s]+'
    non_alpha_numeric_pattern = "[^a-zA-Z0-9]"
    sequence_pattern = r"(.)\1\1+"
    sequence_replace_pattern = r"\1\1"

    for tweet in text_data:
        tweet = tweet.lower()
        
        # Replace URLs with 'URL'
        tweet = re.sub(url_pattern, ' URL', tweet)

        # Replace emojis with 'EMOJI_<meaning>'
        for emoji, meaning in emojis.items():
            tweet = tweet.replace(emoji, f"EMOJI_{meaning}")    

        # Replace @username with 'USER'
        tweet = re.sub(user_pattern, ' USER', tweet)     

        # Replace non-alphanumeric characters with space
        tweet = re.sub(non_alpha_numeric_pattern, " ", tweet)

        # Replace sequences of repeated characters with two of the same
        tweet = re.sub(sequence_pattern, sequence_replace_pattern, tweet)

        tweet_words = []
        for word in tweet.split():
            # Exclude stopwords and words with a single character
            if len(word) > 1:
                # Lemmatize word
                word = lemmatizer.lemmatize(word)
                tweet_words.append(word)
                all_words.append(word)
        
        processed_text.append(' '.join(tweet_words))

    if threshold:
        # Remove words with very low frequency in dataset
        word_counts = Counter(all_words)
        filtered_text = [' '.join(word for word in sentence.split() if word_counts[word] > threshold) 
                    for sentence in processed_text]
        return filtered_text
    
    return processed_text