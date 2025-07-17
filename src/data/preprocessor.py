import re
from bs4 import BeautifulSoup

def clean_text(text: str) -> str:
    """
    Performs basic cleaning on text data:
    - Removes HTML tags.
    - Removes URLs.
    - Removes non-alphanumeric characters (keeping spaces).
    - Removes numbers.
    - Converts text to lowercase.
    - Removes extra whitespace.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs

    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove URLs (http, https, www variations)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)

    # Remove special characters and numbers, keep spaces
    # This keeps only letters and spaces. Adjust if you need to keep punctuation.
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace (tabs, multiple spaces, newlines) and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Example usage (this part will only run if you execute preprocessor.py directly)
if __name__ == "__main__":
    sample_text_1 = "This is a <b>test</b> with a URL: https://example.com and some numbers 123. Also an email test@example.org. And! Punctuation?"
    cleaned_1 = clean_text(sample_text_1)
    print(f"Original 1: {sample_text_1}")
    print(f"Cleaned 1: {cleaned_1}")
    # Expected: "this is a test with a url and some numbers also an email and punctuation"

    sample_text_2 = "  Another example \n with   lots of   spaces and weird -- characters! "
    cleaned_2 = clean_text(sample_text_2)
    print(f"\nOriginal 2: {sample_text_2}")
    print(f"Cleaned 2: {cleaned_2}")
    # Expected: "another example with lots of spaces and weird characters"

    sample_text_3 = 123 # Non-string input
    cleaned_3 = clean_text(sample_text_3)
    print(f"\nOriginal 3: {sample_text_3}")
    print(f"Cleaned 3: '{cleaned_3}'")
    # Expected: Original 3: 123 Cleaned 3: ''