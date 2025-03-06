"""
Utility for parsing and processing Markdown content.
"""

import re
from typing import List, Dict, Any, Union, Tuple, Optional
import markdown
from bs4 import BeautifulSoup


class MarkdownParser:
    """Markdown content parsing and processing utility."""
    
    def __init__(self):
        """Initialize the Markdown parser."""
        self.md = markdown.Markdown(extensions=['extra', 'toc', 'codehilite', 'fenced_code'])
    
    def md_to_html(self, md_content: str) -> str:
        """
        Convert Markdown content to HTML.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            HTML version of the content
        """
        self.md.reset()
        return self.md.convert(md_content)
    
    def extract_headers(self, md_content: str) -> List[Dict[str, Any]]:
        """
        Extract headers from Markdown content with their levels.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            List of dictionaries with header text and level
        """
        headers = []
        lines = md_content.splitlines()
        
        for line in lines:
            header_match = re.match(r'^(#+)\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                text = header_match.group(2).strip()
                headers.append({
                    'level': level,
                    'text': text
                })
        
        return headers
    
    def extract_code_blocks(self, md_content: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from Markdown content.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            List of dictionaries with code content and language
        """
        code_blocks = []
        pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        
        for match in pattern.finditer(md_content):
            language = match.group(1) or "text"
            code = match.group(2)
            code_blocks.append({
                'language': language,
                'code': code
            })
        
        return code_blocks
    
    def extract_lists(self, md_content: str) -> List[Dict[str, Any]]:
        """
        Extract lists from Markdown content.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            List of dictionaries with list items and type
        """
        # Convert to HTML and parse with BeautifulSoup for accurate list extraction
        html = self.md_to_html(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        
        lists = []
        
        # Find all ul and ol elements
        for list_elem in soup.find_all(['ul', 'ol']):
            list_type = 'unordered' if list_elem.name == 'ul' else 'ordered'
            items = [li.get_text() for li in list_elem.find_all('li', recursive=False)]
            
            # Check for nested lists
            nested_items = []
            for li in list_elem.find_all('li', recursive=False):
                nested_list = li.find(['ul', 'ol'])
                if nested_list:
                    nested_type = 'unordered' if nested_list.name == 'ul' else 'ordered'
                    nested_items.append({
                        'parent': li.get_text().replace(nested_list.get_text(), '').strip(),
                        'type': nested_type,
                        'items': [nested_li.get_text() for nested_li in nested_list.find_all('li')]
                    })
            
            lists.append({
                'type': list_type,
                'items': items,
                'nested': nested_items if nested_items else None
            })
        
        return lists
    
    def extract_tables(self, md_content: str) -> List[Dict[str, Any]]:
        """
        Extract tables from Markdown content.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            List of dictionaries with table headers and rows
        """
        html = self.md_to_html(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        
        tables = []
        
        for table in soup.find_all('table'):
            headers = []
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all('th')]
            
            rows = []
            body = table.find('tbody')
            if body:
                for tr in body.find_all('tr'):
                    rows.append([td.get_text().strip() for td in tr.find_all('td')])
            
            tables.append({
                'headers': headers,
                'rows': rows
            })
        
        return tables
    
    def extract_links(self, md_content: str) -> List[Dict[str, str]]:
        """
        Extract links from Markdown content.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            List of dictionaries with link text and URLs
        """
        links = []
        pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        
        for match in pattern.finditer(md_content):
            text = match.group(1)
            url = match.group(2)
            links.append({
                'text': text,
                'url': url
            })
        
        return links
    
    def extract_images(self, md_content: str) -> List[Dict[str, str]]:
        """
        Extract images from Markdown content.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            List of dictionaries with image alt text and source URLs
        """
        images = []
        pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        
        for match in pattern.finditer(md_content):
            alt_text = match.group(1)
            src = match.group(2)
            images.append({
                'alt_text': alt_text,
                'src': src
            })
        
        return images
    
    def md_to_slides_content(self, md_content: str) -> List[Dict[str, Any]]:
        """
        Convert Markdown content to a structured format suitable for slide creation.
        
        Args:
            md_content: Markdown content as a string
            
        Returns:
            List of slide content dictionaries
        """
        headers = self.extract_headers(md_content)
        slides = []
        current_slide = None
        
        # Use headers as slide boundaries
        for header in headers:
            if header['level'] == 1 or header['level'] == 2:
                if current_slide:
                    slides.append(current_slide)
                current_slide = {
                    'title': header['text'],
                    'content': []
                }
            elif current_slide:
                current_slide['content'].append({
                    'type': 'subheading',
                    'text': header['text'],
                    'level': header['level']
                })
        
        # Add the last slide if there is one
        if current_slide:
            slides.append(current_slide)
            
        # Now process the content between headers
        sections = re.split(r'^#+\s+.+$', md_content, flags=re.MULTILINE)
        sections = sections[1:] if len(sections) > len(slides) else sections
        
        for i, section in enumerate(sections):
            if i < len(slides):
                # Extract lists
                for list_match in self.extract_lists(section):
                    slides[i]['content'].append({
                        'type': 'list',
                        'list_type': list_match['type'],
                        'items': list_match['items']
                    })
                
                # Extract code blocks
                for code_block in self.extract_code_blocks(section):
                    slides[i]['content'].append({
                        'type': 'code',
                        'language': code_block['language'],
                        'code': code_block['code']
                    })
                
                # Extract images
                for image in self.extract_images(section):
                    slides[i]['content'].append({
                        'type': 'image',
                        'alt': image['alt_text'],
                        'src': image['src']
                    })
                
                # Extract tables
                for table in self.extract_tables(section):
                    slides[i]['content'].append({
                        'type': 'table',
                        'headers': table['headers'],
                        'rows': table['rows']
                    })
                
                # Add any remaining text as paragraphs
                # Remove already processed elements and get remaining text
                clean_section = re.sub(r'```.*?```', '', section, flags=re.DOTALL)
                clean_section = re.sub(r'!\[.*?\]\(.*?\)', '', clean_section)
                clean_section = re.sub(r'\|.*?\|', '', clean_section)
                
                paragraphs = [p.strip() for p in clean_section.split('\n\n') if p.strip()]
                for para in paragraphs:
                    if not re.match(r'^[-*+]\s+', para) and para.strip():
                        slides[i]['content'].append({
                            'type': 'paragraph',
                            'text': para.strip()
                        })
        
        return slides
