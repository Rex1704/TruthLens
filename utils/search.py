from tavily import TavilyClient
from config.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

def get_search_client():
    try:
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is not set in environment variables.")
        client = TavilyClient(api_key=TAVILY_API_KEY)
        return client
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Tavily client: {e}")
    
def web_search(client, query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    try:
        response = client.search(
            query=query,
            max_results=max_results,
        )
        results = response.get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results
        ]
    except Exception as e:
        return [{"title": "Search Error", "url": "", "content": f"Web search failed: {e}"}]
    
def format_search_results(results: list[dict]) -> str:
    try:
        if not results:
            return "No web search results found for query."
        
        formatted = []
        for i, result in enumerate(results):
            entry = (
                f"[Web Source {i+1}]: {result['title']}\n"
                f"URL: {result['url']}\n"
                f"Content: {result['content']}"
            )
            formatted.append(entry)
        
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Failed to format search results: {e}"

def get_web_context(client, query: str) -> str:
    try:
        results = web_search(client, query)
        return format_search_results(results)
    except Exception as e:
        return f"Web search pipeline error: {e}"