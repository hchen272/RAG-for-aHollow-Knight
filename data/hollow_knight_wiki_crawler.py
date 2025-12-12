import requests
import time
import json
import os
from urllib.parse import quote
from bs4 import BeautifulSoup
import re

class HollowKnightFandomCrawler:
    def __init__(self, base_delay=0.5):
        """
        Initialize the crawler for Hollow Knight Wiki on Fandom
        """
        self.base_url = "https://hollowknight.fandom.com/api.php"
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (RAG Corpus Builder; Educational Purpose)'
        })

    def get_all_pages(self, limit=2000):
        """
        Get all page titles from the Fandom Wiki using allpages API
        """
        all_pages = []
        continue_param = {}
        
        print("Starting to fetch page list from Fandom...")
        
        while len(all_pages) < limit:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'allpages',
                'aplimit': 'max',
                'apnamespace': 0,  # Main namespace only
                **continue_param
            }
            
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'query' in data:
                    batch_pages = data['query']['allpages']
                    if not batch_pages:
                        break
                    
                    page_titles = [page['title'] for page in batch_pages]
                    all_pages.extend(page_titles)
                    print(f"Retrieved {len(all_pages)} page titles...")
                
                # Check if there are more pages
                if 'continue' in data and len(all_pages) < limit:
                    continue_param = {'apcontinue': data['continue']['apcontinue']}
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page list: {e}")
                break
                
            time.sleep(self.base_delay)
        
        return all_pages[:limit]

    def get_page_content(self, title):
        """
        Extract plain text content from a Fandom page
        """
        # First try with query API to get extracts
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|info',
            'explaintext': True,
            'exsectionformat': 'plain',
            'inprop': 'url'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                # Check if page exists
                if page_id != '-1':
                    # Try to get extract content
                    if 'extract' in page_data and page_data['extract'].strip():
                        content = page_data['extract'].strip()
                        fullurl = page_data.get('fullurl', f"https://hollowknight.fandom.com/wiki/{quote(title.replace(' ', '_'))}")
                        
                        return {
                            'title': page_data['title'],
                            'content': content,
                            'url': fullurl,
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'content_length': len(content),
                            'method': 'extract'
                        }
                    
                    # If extract is empty, try using parse API
                    return self.get_page_content_parse(title)
                    
        except requests.exceptions.RequestException as e:
            print(f"Error fetching content for '{title}': {e}")
            
        return None

    def get_page_content_parse(self, title):
        """
        Alternative method using parse API to get page content
        """
        params = {
            'action': 'parse',
            'format': 'json',
            'page': title,
            'prop': 'text',
            'contentmodel': 'wikitext'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'parse' in data:
                parse_data = data['parse']
                html_content = parse_data.get('text', {}).get('*', '')
                
                # Clean HTML content
                clean_text = self.clean_fandom_html_content(html_content)
                
                if clean_text:
                    return {
                        'title': parse_data['title'],
                        'content': clean_text,
                        'url': f"https://hollowknight.fandom.com/wiki/{quote(title.replace(' ', '_'))}",
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'content_length': len(clean_text),
                        'method': 'parse_clean'
                    }
                    
        except requests.exceptions.RequestException as e:
            print(f"Error parsing content for '{title}': {e}")
            
        return None

    def clean_fandom_html_content(self, html_content):
        """
        Extract clean text from Fandom HTML content
        """
        if not html_content:
            return None
            
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove Fandom-specific unwanted elements
        unwanted_selectors = [
            'script', 'style', 'nav', 'header', 'footer', 
            'table', 'sup', '.navbox', '.reference', 
            '.mw-editsection', '.pi-data', '.pi-horizontal-group',
            '.portable-infobox', '.wikia-menu-button', '.wds-dropdown',
            '.global-navigation', '.fandom-community-header',
            '.page-sidebar', '.page__right-rail', '.ads'
        ]
        
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Find main content area (Fandom specific)
        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if not content_div:
            # Try alternative content containers
            content_div = soup.find('div', {'id': 'content'}) or soup
        
        # Extract text from relevant elements
        text_elements = []
        for element in content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'td']):
            text = element.get_text().strip()
            
            # Clean the text
            text = re.sub(r'\[\d+\]', '', text)  # Remove reference markers
            text = re.sub(r'\[edit\]', '', text)  # Remove edit markers
            text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
            
            # Only keep meaningful text
            if (len(text) > 30 and 
                not any(x in text for x in [
                    'Navigation menu', 'Main article', 'Category:',
                    'Fandom', 'Shop', 'Subscribe', 'Follow'
                ])):
                text_elements.append(text)
        
        return '\n'.join(text_elements)

    def test_api_connection(self):
        """
        Test the API connection and get site info
        """
        params = {
            'action': 'query',
            'meta': 'siteinfo',
            'format': 'json'
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data:
                siteinfo = data['query']['general']
                print(f"Site Name: {siteinfo.get('sitename', 'Unknown')}")
                print(f"Base URL: {siteinfo.get('base', 'Unknown')}")
                print(f"API URL: {self.base_url}")
                return True
                
        except requests.exceptions.RequestException as e:
            print(f"API connection test failed: {e}")
            
        return False

    def crawl_fandom_wiki(self, output_filename="hollow_knight_fandom_corpus.json", max_pages=None):
        """
        Main method to crawl the Fandom Wiki and save to file
        """
        print("Starting Hollow Knight Fandom Wiki crawl...")
        
        # Test API connection first
        if not self.test_api_connection():
            print("Failed to connect to Fandom API. Please check the URL.")
            return None
        
        # Get all page titles
        all_titles = self.get_all_pages()
        
        if max_pages:
            all_titles = all_titles[:max_pages]
            print(f"Limited to first {max_pages} pages")
        
        print(f"Preparing to crawl {len(all_titles)} pages...")
        
        wiki_corpus = []
        success_count = 0
        failed_pages = []
        
        for i, title in enumerate(all_titles):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{len(all_titles)} (Success: {success_count})")
            
            content = self.get_page_content(title)
            if content:
                wiki_corpus.append(content)
                success_count += 1
                print(f"✓ '{title}' - {content['content_length']} chars (method: {content.get('method', 'unknown')})")
            else:
                failed_pages.append(title)
                print(f"✗ Failed: '{title}'")
            
            # Respectful delay between requests
            time.sleep(self.base_delay)
        
        # Save to current script directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
        output_path = os.path.join(script_dir, output_filename)
        
        # Save corpus as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_pages_attempted': len(all_titles) if not max_pages else min(len(all_titles), max_pages),
                    'successful_pages': success_count,
                    'failed_pages': len(failed_pages),
                    'crawl_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'Hollow Knight Fandom Wiki',
                    'api_url': self.base_url
                },
                'failed_pages_list': failed_pages,
                'pages': wiki_corpus
            }, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\nCrawl completed!")
        print(f"Successful: {success_count}/{len(all_titles)} pages")
        print(f"Failed: {len(failed_pages)} pages")
        print(f"Output saved to: {output_path}")
        
        if failed_pages and len(failed_pages) < 20:
            print(f"Failed pages: {failed_pages}")
        elif failed_pages:
            print(f"First 20 failed pages: {failed_pages[:20]}")
        
        return output_path

    def create_rag_optimized_corpus(self, input_file, output_file):
        """
        Convert raw data to RAG-optimized format with chunked content
        """
        # Use absolute path for input file
        script_dir = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
        input_path = os.path.join(script_dir, input_file)
        
        print(f"Reading from: {input_path}")
        
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}")
            return None
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rag_documents = []
        
        for page in data['pages']:
            # Split content into paragraphs
            paragraphs = [p.strip() for p in page['content'].split('\n') if p.strip()]
            
            # Further split long paragraphs
            chunks = []
            for paragraph in paragraphs:
                if len(paragraph) > 500:
                    # Split long paragraphs by sentences
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 500:
                            current_chunk += " " + sentence
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                else:
                    chunks.append(paragraph)
            
            # Create RAG documents
            for i, chunk in enumerate(chunks):
                if len(chunk) > 50:  # Only keep meaningful chunks
                    rag_documents.append({
                        'id': f"{page['title']}_chunk_{i}",
                        'title': page['title'],
                        'content': chunk,
                        'url': page['url'],
                        'source': 'hollow_knight_fandom_wiki',
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'timestamp': page.get('timestamp', '')
                    })
        
        # Save RAG-optimized data
        output_path = os.path.join(script_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rag_documents, f, ensure_ascii=False, indent=2)
        
        print(f"RAG-optimized data saved to: {output_path}")
        print(f"Created {len(rag_documents)} document chunks from {len(data['pages'])} pages")
        
        return rag_documents

def main():
    """
    Main execution function
    """
    crawler = HollowKnightFandomCrawler(base_delay=0.5)
    
    # Step 1: Crawl all pages from Fandom
    print("=== STEP 1: Crawling Fandom Wiki Pages ===")
    output_file = "hollow_knight_fandom_corpus.json"
    
    # Start with a small number for testing, then set to None for all pages
    crawled_file_path = crawler.crawl_fandom_wiki(
        output_filename=output_file,
        max_pages=None  # Start with 20 pages for testing, change to None for all pages
    )
    
    # Step 2: Create RAG-optimized corpus
    if crawled_file_path and os.path.exists(crawled_file_path):
        print("\n=== STEP 2: Creating RAG-Optimized Corpus ===")
        crawler.create_rag_optimized_corpus(
            output_file,
            "hollow_knight_fandom_rag_optimized.json"
        )
    else:
        print(f"Error: Crawled file not found at {crawled_file_path}")
    
    print("\n=== PROCESS COMPLETED ===")

if __name__ == "__main__":
    main()