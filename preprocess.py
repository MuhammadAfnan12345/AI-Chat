from bs4 import BeautifulSoup
import re
import json
import os

def clean_html(raw_html):
    """
    Remove HTML tags, scripts, and extra whitespaces.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text, max_chars=500):
    """
    Split cleaned text into smaller chunks.
    """
    sentences = re.split(r'\.\s+', text)
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chars:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "

    if chunk:
        chunks.append(chunk.strip())

    return chunks


def preprocess_text_file(input_txt_path, output_json_path):
    """
    Clean and chunk the text, save as JSON.
    """
    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    cleaned_text = clean_html(raw_html)
    chunks = split_into_chunks(cleaned_text)

    result = [{"chunk_id": i, "content": chunk} for i, chunk in enumerate(chunks)]

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"Processed {input_txt_path} ➜ {output_json_path} ({len(result)} chunks)")


# Example usage
if __name__ == "__main__":
    os.makedirs("cleaned_data", exist_ok=True)

    preprocess_text_file("fast_site.txt", "cleaned_data/fast_cleaned.json")
    preprocess_text_file("kdd_lab.txt", "cleaned_data/kdd_cleaned.json")
