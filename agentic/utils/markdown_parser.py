"""
Utility class for parsing and analyzing Markdown content.
"""

import re
from typing import List, Dict, Any, Union, Tuple, Optional
import markdown
from bs4 import BeautifulSoup


class MarkdownParser:
    """
    A utility class that provides methods for parsing and analyzing Markdown content.
    """
    
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
    
    def extract_headers(self, markdown_content):
        """Extract headers from Markdown content."""
        headers = []
        for line in markdown_content.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                # Count the number of # to determine the level
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                # Extract the header text without the # symbols
                header_text = line[level:].strip()
                
                headers.append({
                    'level': level,
                    'text': header_text
                })
        
        return headers
    
    def extract_lists(self, markdown_content):
        """Extract lists from Markdown content."""
        lists = []
        current_list = None
        
        for line in markdown_content.split('\n'):
            line = line.strip()
            
            # Check for list items
            if line.startswith(('- ', '* ', '+ ', '1. ', '2. ')):
                if current_list is None:
                    # Start a new list
                    list_type = 'unordered' if line.startswith(('- ', '* ', '+ ')) else 'ordered'
                    current_list = {
                        'type': 'list',
                        'list_type': list_type,
                        'items': []
                    }
                
                # Extract the list item text
                if line.startswith(('- ', '* ', '+ ')):
                    item_text = line[2:].strip()
                else:  # ordered list
                    parts = line.split('. ', 1)
                    if len(parts) > 1:
                        item_text = parts[1].strip()
                    else:
                        item_text = ''
                
                current_list['items'].append(item_text)
            else:
                # End of the list
                if current_list is not None:
                    lists.append(current_list)
                    current_list = None
        
        # Add the last list if there is one
        if current_list is not None:
            lists.append(current_list)
        
        return lists
    
    def extract_code_blocks(self, markdown_content):
        """Extract code blocks from Markdown content."""
        code_blocks = []
        in_code_block = False
        current_block = None
        language = ''
        
        for line in markdown_content.split('\n'):
            if line.startswith('```'):
                if not in_code_block:
                    # Start of a code block
                    in_code_block = True
                    language = line[3:].strip()
                    current_block = {
                        'type': 'code',
                        'language': language,
                        'code': ''
                    }
                else:
                    # End of a code block
                    in_code_block = False
                    code_blocks.append(current_block)
                    current_block = None
            elif in_code_block and current_block is not None:
                # Add the line to the current code block
                if current_block['code']:
                    current_block['code'] += '\n'
                current_block['code'] += line
        
        return code_blocks
    
    def extract_tables(self, markdown_content):
        """Extract tables from Markdown content."""
        tables = []
        in_table = False
        current_table = None
        
        for line in markdown_content.split('\n'):
            line = line.strip()
            
            # Check for table rows
            if '|' in line:
                if not in_table:
                    # Start of a table
                    in_table = True
                    current_table = {
                        'type': 'table',
                        'headers': [],
                        'rows': []
                    }
                    
                    # Extract the headers
                    cells = [cell.strip() for cell in line.split('|')]
                    # Remove empty cells at the beginning and end
                    if cells and not cells[0]:
                        cells = cells[1:]
                    if cells and not cells[-1]:
                        cells = cells[:-1]
                    
                    current_table['headers'] = cells
                elif line.replace('-', '').replace('|', '').replace(':', '').strip() == '':
                    # This is the separator line, skip it
                    continue
                else:
                    # This is a data row
                    cells = [cell.strip() for cell in line.split('|')]
                    # Remove empty cells at the beginning and end
                    if cells and not cells[0]:
                        cells = cells[1:]
                    if cells and not cells[-1]:
                        cells = cells[:-1]
                    
                    current_table['rows'].append(cells)
            else:
                # End of the table
                if in_table and current_table is not None:
                    tables.append(current_table)
                    current_table = None
                    in_table = False
        
        # Add the last table if there is one
        if in_table and current_table is not None:
            tables.append(current_table)
        
        return tables
    
    def extract_images(self, markdown_content):
        """Extract images from Markdown content."""
        images = []
        
        # Regular expression to match Markdown image syntax
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        
        for match in re.finditer(image_pattern, markdown_content):
            alt_text = match.group(1)
            image_url = match.group(2)
            
            images.append({
                'type': 'image',
                'alt': alt_text,
                'src': image_url
            })
        
        return images
    
    def md_to_slides_content(self, markdown_content):
        """Convert Markdown content to slides structure."""
        headers = self.extract_headers(markdown_content)
        if not headers:
            # If no headers, treat as a single slide
            return [{
                'title': 'Slide',
                'content': self._extract_content_between_headers(markdown_content, 0, len(markdown_content))
            }]
        
        # Split the content into slides based on h1 headers
        slides_data = []
        h1_indexes = []
        
        # Find all h1 headers
        lines = markdown_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('# '):
                h1_indexes.append(i)
        
        # Add an end index
        h1_indexes.append(len(lines))
        
        # Create slides from h1 sections
        for i in range(len(h1_indexes) - 1):
            start_idx = h1_indexes[i]
            end_idx = h1_indexes[i+1]
            
            # Get the slide title (h1 header)
            title = lines[start_idx][2:].strip()
            
            # Extract content between this h1 and the next one
            content_text = '\n'.join(lines[start_idx+1:end_idx])
            content = []
            
            # Extract lists
            lists = self.extract_lists(content_text)
            content.extend(lists)
            
            # Extract code blocks
            code_blocks = self.extract_code_blocks(content_text)
            content.extend(code_blocks)
            
            # Extract images
            images = self.extract_images(content_text)
            content.extend(images)
            
            # Extract tables
            tables = self.extract_tables(content_text)
            content.extend(tables)
            
            # Add subheadings (h2, h3, etc.)
            subheaders = [h for h in self.extract_headers(content_text) if h['level'] > 1]
            for header in subheaders:
                content.append({
                    'type': 'subheading',
                    'level': header['level'],
                    'text': header['text']
                })
            
            slides_data.append({
                'title': title,
                'content': content
            })
        
        return slides_data
    
    def _extract_content_between_headers(self, text, start, end):
        """Helper method to extract content between headers."""
        content = []
        
        # Extract the text between start and end
        content_text = text[start:end]
        
        # Extract lists, code blocks, images, and tables
        lists = self.extract_lists(content_text)
        content.extend(lists)
        
        code_blocks = self.extract_code_blocks(content_text)
        content.extend(code_blocks)
        
        images = self.extract_images(content_text)
        content.extend(images)
        
        tables = self.extract_tables(content_text)
        content.extend(tables)
        
        return content
