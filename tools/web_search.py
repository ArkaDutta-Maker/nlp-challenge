"""
Web Search Tool for fetching content from hyperlinks and search queries.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict, Any
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchTool:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.timeout = 10
        self.max_content_length = 5000  # Limit content to avoid token overflow

    def extract_hyperlinks(self, text: str) -> List[str]:
        """Extract all hyperlinks from text."""
        # More comprehensive pattern for URLs - handles all URL components properly
        url_pattern = r'https?://[^\s<>\[\]{}|\\^`"]+'
        
        urls = re.findall(url_pattern, text)
        logger.info(f"Found {len(urls)} URLs using regex: {urls}")
        
        # Clean up URLs that might have ended with punctuation
        cleaned_urls = []
        for url in urls:
            # Remove trailing punctuation that's not part of URL
            url = re.sub(r'[.!?;,]+$', '', url)
            cleaned_urls.append(url)
        
        # Also look for markdown links [text](url)
        markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        markdown_matches = re.findall(markdown_pattern, text)
        for _, url in markdown_matches:
            if url.startswith('http'):
                cleaned_urls.append(url)
                logger.info(f"Found markdown URL: {url}")
        
        # Remove duplicates and return
        unique_urls = list(set(cleaned_urls))
        logger.info(f"Final unique URLs: {unique_urls}")
        return unique_urls

    def fetch_webpage_content(self, url: str) -> Dict[str, Any]:
        """Fetch and extract content from a webpage."""
        try:
            logger.info(f"Fetching content from: {url}")
            
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {
                    "url": url,
                    "success": False,
                    "error": "Invalid URL format",
                    "content": ""
                }
            
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.string.strip() if title else "No title"
            
            # Extract main content (try common content selectors)
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post', '.entry-content', '.article-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no specific content found, get body text
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator=' ', strip=True)
            
            # Clean and limit content
            content_text = re.sub(r'\s+', ' ', content_text).strip()
            if len(content_text) > self.max_content_length:
                content_text = content_text[:self.max_content_length] + "..."
            
            return {
                "url": url,
                "success": True,
                "title": title_text,
                "content": content_text,
                "length": len(content_text)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {str(e)}")
            return {
                "url": url,
                "success": False,
                "error": f"Request failed: {str(e)}",
                "content": ""
            }
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            return {
                "url": url,
                "success": False,
                "error": f"Processing error: {str(e)}",
                "content": ""
            }

    def search_web_content(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Search and fetch content from multiple URLs."""
        results = []
        
        for url in urls[:3]:  # Limit to first 3 URLs to avoid timeout
            result = self.fetch_webpage_content(url)
            results.append(result)
            
            # Add small delay between requests to be respectful
            time.sleep(0.5)
        
        return results

    def format_web_results(self, web_results: List[Dict[str, Any]]) -> str:
        """Format web search results for display."""
        if not web_results:
            return ""
        
        formatted_results = []
        for result in web_results:
            if result["success"] and result["content"]:
                # Include more content for better context (up to 2000 chars)
                content_preview = result['content'][:2000]
                if len(result['content']) > 2000:
                    content_preview += "..."
                    
                formatted_results.append(
                    f"**Page Title:** {result['title']}\n"
                    f"**URL:** {result['url']}\n"
                    f"**Page Content:**\n{content_preview}\n"
                )
        
        if formatted_results:
            return "\n\n".join(formatted_results)
        else:
            return ""

# Tool functions for agent integration
def extract_and_search_hyperlinks(text: str) -> str:
    """
    Extract hyperlinks from text and fetch their content.
    Returns formatted web content that can be appended to answers.
    """
    tool = WebSearchTool()
    
    logger.info(f"Searching for hyperlinks in text (length: {len(text)})")
    logger.info(f"Text preview: {text[:200]}...")
    
    # Extract URLs from the text
    urls = tool.extract_hyperlinks(text)
    
    if not urls:
        logger.info("No URLs found in text")
        return ""
    
    logger.info(f"Found {len(urls)} URLs to fetch: {urls}")
    
    # Fetch content from URLs
    web_results = tool.search_web_content(urls)
    
    # Format results
    formatted_result = tool.format_web_results(web_results)
    logger.info(f"Formatted web results length: {len(formatted_result)}")
    
    return formatted_result

def web_search_action(urls: List[str]) -> Dict[str, Any]:
    """
    Action function for web searching specific URLs.
    Returns structured data for the agent workflow.
    """
    tool = WebSearchTool()
    
    web_results = tool.search_web_content(urls)
    
    # Count successful fetches
    successful_results = [r for r in web_results if r["success"]]
    
    return {
        "web_results": web_results,
        "successful_count": len(successful_results),
        "total_content_length": sum(len(r.get("content", "")) for r in successful_results),
        "formatted_content": tool.format_web_results(web_results)
    }

# Test function
if __name__ == "__main__":
    # Test the web search tool
    tool = WebSearchTool()
    
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.example.com"
    ]
    
    results = tool.search_web_content(test_urls)
    print("Web Search Results:")
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Title: {result['title']}")
            print(f"Content (first 200 chars): {result['content'][:200]}...")
        else:
            print(f"Error: {result['error']}")
        print("-" * 50)