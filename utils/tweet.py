import re
import requests

TWEET_URL_PATTERN = re.compile(
    r"https?://(www\.)?(twitter\.com|x\.com)/\w+/status/\d+"
)


def is_tweet_url(text: str) -> bool:
    """Returns True if the input looks like a Twitter/X URL."""
    return bool(TWEET_URL_PATTERN.search(text.strip()))


def normalize_tweet_url(url: str) -> str:
    return re.sub(r"https?://(www\.)?x\.com", "https://twitter.com", url.strip())


def fetch_tweet_text(url: str) -> dict:
    try:
        clean_url = normalize_tweet_url(url)

        oembed_endpoint = "https://publish.twitter.com/oembed"
        params = {
            "url": clean_url,
            "omit_script": True,
            "maxwidth": 550,
        }

        response = requests.get(oembed_endpoint, params=params, timeout=8)

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Twitter oEmbed API returned status {response.status_code}. "
                         "The tweet may be private, deleted, or the URL is invalid.",
            }

        data = response.json()
        raw_html = data.get("html", "")
        author = data.get("author_name", "Unknown")

        tweet_text = _extract_text_from_oembed_html(raw_html)

        return {
            "success": True,
            "text": tweet_text,
            "author": author,
            "html": raw_html,
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Twitter may be slow — try again."}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not connect to Twitter. Check your internet connection."}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error fetching tweet: {e}"}


def _extract_text_from_oembed_html(html: str) -> str:
    try:
        text = re.sub(r"<a[^>]*>", "", html)
        text = re.sub(r"</a>", "", text)

        text = re.sub(r"<[^>]+>", " ", text)

        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")

        text = re.sub(r"\s+", " ", text).strip()

        text = re.sub(r"—\s+.+?\(@\w+\).+$", "", text).strip()

        return text

    except Exception:
        return html 