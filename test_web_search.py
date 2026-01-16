"""
Test script for the web search functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.web_search import WebSearchTool, extract_and_search_hyperlinks

def test_web_search():
    """Test the web search tool functionality"""
    
    print("🧪 Testing Web Search Tool...")
    
    # Test 1: Complex URL from the user's example
    complex_text = """
    The QR code on page 45 is associated with a link to HCL Tech's digital operations webpage, specifically highlighting customer stories. The full URL is https://www.hcltech.com/digital-operations?utm_source=qr-code&utm_medium=scan&utm_campaign=FY26_CMO_REP_Annual-Report_072025#customer-stories. This information is available on page 45, in the context of a description about automating the payor/claims process for a client.
    """
    
    tool = WebSearchTool()
    urls = tool.extract_hyperlinks(complex_text)
    print(f"\n✅ Extracted URLs from complex text: {urls}")
    print(f"   Expected: https://www.hcltech.com/digital-operations?utm_source=qr-code&utm_medium=scan&utm_campaign=FY26_CMO_REP_Annual-Report_072025#customer-stories")
    
    # Test 2: Extract and search hyperlinks function with the complex URL
    print(f"\n🌐 Testing extract_and_search_hyperlinks with complex URL...")
    web_content = extract_and_search_hyperlinks(complex_text)
    
    if web_content:
        print(f"✅ Web content retrieved (length: {len(web_content)})")
        print(f"Content preview: {web_content[:300]}...")
    else:
        print("⚠️ No web content retrieved")
    
    # Test 3: Multiple URL types
    multi_url_text = """
    Visit https://example.com/simple for basic info.
    Complex URL: https://www.hcltech.com/digital-operations?utm_source=qr-code&utm_medium=scan&utm_campaign=FY26_CMO_REP_Annual-Report_072025#customer-stories
    Also check [Python docs](https://docs.python.org/3/) in markdown format.
    """
    
    urls_multi = tool.extract_hyperlinks(multi_url_text)
    print(f"\n✅ Multiple URL extraction test: {len(urls_multi)} URLs found")
    for i, url in enumerate(urls_multi, 1):
        print(f"   {i}. {url}")
    
    print("\n🎉 Web search tool test completed!")

if __name__ == "__main__":
    test_web_search()