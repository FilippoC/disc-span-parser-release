import unicodedata

def is_punct(token):
    return all(unicodedata.category(char).startswith('P') for char in token)