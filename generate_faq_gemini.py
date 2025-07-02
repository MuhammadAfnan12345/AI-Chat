# generate_faq_gemini.py
import json
import time
from gemini_api import generate_gemini_response

def load_chunks(files):
    """
    Read all 'content' fields from a list of cleaned-data JSON files.
    """
    chunks = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            chunks.extend([entry["content"] for entry in data])
    return chunks

def generate_qa(chunk):
    """
    Build the Q&A prompt for a single chunk and call Gemini.
    """
    prompt = (
            "You are a university assistant. Based on the following website content, generate exactly 4 helpful and specific question-answer pairs "
            "for students or visitors. Reply ONLY as a JSON array with objects having 'question' and 'answer' keys.\n\n"
            f"Content:\n{chunk}"
        )
    return generate_gemini_response(prompt)

def parse_qa_response(text):
    """
    Parse the JSON text returned by Gemini into Python list.
    """
    cleaned = text.strip()
    # strip code fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return []

def generate_faq():
    input_files = [
        "cleaned_data/fast_cleaned.json",
        "cleaned_data/kdd_cleaned.json"
    ]
    output_file = "faq.json"

    print("🔄 Loading chunks…")
    chunks = load_chunks(input_files)
    print(f"📄 Total chunks: {len(chunks)}")

    faq_list = []
    for i, chunk in enumerate(chunks, start=1):
        print(f"✨ Chunk {i}/{len(chunks)}")
        resp_text = generate_qa(chunk)
        qa_pairs = parse_qa_response(resp_text)
        if qa_pairs:
            faq_list.extend(qa_pairs)
        else:
            print("⚠️  No valid JSON Q&A in response.")
        time.sleep(1.5)  # to avoid rate limits

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(faq_list, out, indent=4)

    print(f"\n✅ Done! ({len(faq_list)} Q&A pairs) saved to {output_file}")

if __name__ == "__main__":
    generate_faq()