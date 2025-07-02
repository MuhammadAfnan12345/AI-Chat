# scraper.py
import requests
from bs4 import BeautifulSoup

def fetch_and_save(url: str, filename: str, max_chars: int = 20000):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text[:max_chars])
        print(f"✅ Saved {filename} with {len(text[:max_chars])} characters.")
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {str(e)}")

if __name__ == "__main__":
    fetch_and_save("https://www.nu.edu.pk", "fast_site.txt")
