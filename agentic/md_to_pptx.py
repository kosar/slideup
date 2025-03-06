"""
Markdown to PowerPoint converter module.
"""

from pathlib import Path
from typing import Union, Optional, Dict, List, Any, Tuple
import re
import os

from .utils.ppt_manager import PPTManager
from .utils.markdown_parser import MarkdownParser
from .utils.file_handler import FileHandler
from .utils.logger import setup_logger, ErrorHandler


class MarkdownToPPTX:
    """Converter class to transform Markdown content into PowerPoint presentations."""
    
    def __init__(
        self,
        template_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        image_dir: Optional[Union[str, Path]] = None,
        logger_name: str = "md_to_pptx"
    ):
        """
        Initialize the Markdown to PowerPoint converter.
        
        Args:
            template_path: Path to a PowerPoint template file
            output_dir: Directory to save generated presentations
            image_dir: Directory containing images referenced in Markdown
            logger_name: Name for the logger
        """
        self.template_path = template_path
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.image_dir = Path(image_dir) if image_dir else None
        
        # Initialize utilities
        self.ppt = PPTManager(template_path)
        self.md_parser = MarkdownParser()
        self.file_handler = FileHandler()
        
        # Set up logging
        self.logger = setup_logger(logger_name)
        self.error_handler = ErrorHandler(self.logger)
        
    def convert_file(
        self,
        markdown_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
        subtitle: Optional[str] = None
    ) -> str:
        """
        Convert a Markdown file to a PowerPoint presentation.
        
        Args:
            markdown_path: Path to the Markdown file
            output_path: Path where the PowerPoint file will be saved
            title: Optional title for the presentation
            subtitle: Optional subtitle for the presentation
            
        Returns:
            Path to the created PowerPoint file
        """
        try:
            # Read markdown content
            md_content = self.file_handler.read_text(markdown_path)
            
            # Determine output path if not provided
            if output_path is None:
                md_filename = Path(markdown_path).stem
                output_path = self.output_dir / f"{md_filename}.pptx"
            
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert markdown content to PowerPoint
            self.convert_content(
                md_content, 
                output_path, 
                title=title, 
                subtitle=subtitle, 
                md_file_dir=Path(markdown_path).parent
            )
            
            self.logger.info(f"Successfully converted {markdown_path} to {output_path}")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"Failed to convert {markdown_path}"
            self.error_handler.handle_exception(e, error_msg)
            return ""
    
    def convert_content(
        self,
        md_content: str,
        output_path: Union[str, Path],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        md_file_dir: Optional[Path] = None
    ) -> None:
        """
        Convert Markdown content to a PowerPoint presentation.
        
        Args:
            md_content: Markdown content as a string
            output_path: Path where the PowerPoint file will be saved
            title: Optional title for the presentation
            subtitle: Optional subtitle for the presentation
            md_file_dir: Directory of the source Markdown file (for resolving relative paths)
        """
        # Parse markdown content into slides structure
        slides_data = self.md_parser.md_to_slides_content(md_content)
        
        # If no slides were generated, create an empty presentation
        if not slides_data:
            self.ppt.save(output_path)
            self.logger.warning("No slides content found in the Markdown file")
            return
        
        # Create title slide if title is provided or use first header as title
        if title or subtitle:
            self.ppt.add_title_slide(
                title=title or "Presentation", 
                subtitle=subtitle or ""
            )
        elif slides_data and slides_data[0]['title']:
            # Use the first header as the title slide if it's a level 1 header
            first_slide = slides_data[0]
            self.ppt.add_title_slide(
                title=first_slide['title'],
                subtitle=""
            )
            # Remove the first slide from the data to avoid duplication
            slides_data = slides_data[1:]
        
        # Process each slide
        for slide_data in slides_data:
            self._create_slide_from_data(slide_data, md_file_dir)
        
        # Save the presentation
        self.ppt.save(output_path)
    
    def _create_slide_from_data(self, slide_data: Dict[str, Any], md_file_dir: Optional[Path] = None) -> None:
        """
        Create a PowerPoint slide from structured slide data.
        
        Args:
            slide_data: Dictionary containing slide content data
            md_file_dir: Directory of the source Markdown file
        """
        title = slide_data['title']
        content = slide_data.get('content', [])
        
        # Check for content types to determine appropriate slide layout
        has_images = any(item['type'] == 'image' for item in content)
        has_code = any(item['type'] == 'code' for item in content)
        has_table = any(item['type'] == 'table' for item in content)
        has_list = any(item['type'] == 'list' for item in content)
        
        # Create a slide with appropriate layout
        if has_images and (has_code or has_table):
            # Complex slide with image and code/table - use blank layout
            slide = self.ppt.add_slide(6)  # Using blank layout
            # Add title manually
            self.ppt.add_textbox(slide, title, left=0.5, top=0.5, width=9.0, height=0.75, font_size=24)
            top_position = 1.25
        else:
            # Use title and content layout
            slide = self.ppt.add_slide(1)
            if hasattr(slide.shapes, 'title') and slide.shapes.title:
                slide.shapes.title.text = title
            top_position = 1.5  # Start content below title
        
        # Initialize positions for content
        left_position = 0.5
        content_width = 9.0
        
        # Add content to the slide
        for item in content:
            item_type = item.get('type', '')
            
            if item_type == 'list':
                list_items = item.get('items', [])
                if list_items:
                    # Find content placeholder or create textbox
                    text_frame = None
                    for shape in slide.placeholders:
                        if hasattr(shape, 'placeholder_format') and shape.placeholder_format.type == 2:  # Content placeholder
                            text_frame = shape.text_frame
                            text_frame.clear()
                            break
                    
                    if not text_frame:
                        textbox = self.ppt.add_textbox(
                            slide, "", 
                            left=left_position, 
                            top=top_position,
                            width=content_width, 
                            height=len(list_items) * 0.3 + 0.5
                        )
                        text_frame = textbox.text_frame
                    
                    # Add list items
                    for i, list_item in enumerate(list_items):
                        p = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
                        p.text = list_item
                        p.level = 0  # Top-level bullet
                    
                    top_position += len(list_items) * 0.3 + 0.5
            
            elif item_type == 'image':
                image_src = item.get('src', '')
                if image_src:
                    # Resolve image path
                    image_path = self._resolve_image_path(image_src, md_file_dir)
                    if image_path and os.path.exists(image_path):
                        # Calculate image dimensions to maintain aspect ratio
                        width = min(content_width, 5.0)
                        self.ppt.add_picture(
                            slide, image_path,
                            left=left_position + (content_width - width) / 2,  # Center image
                            top=top_position,
                            width=width
                        )
                        top_position += 3.0  # Estimated height, ideally should be calculated based on aspect ratio
            
            elif item_type == 'code':
                code = item.get('code', '')
                if code:
                    code_lines = code.split('\n')
                    height = len(code_lines) * 0.2 + 0.3  # Estimate height based on number of lines
                    
                    # Create textbox for code
                    textbox = self.ppt.add_textbox(
                        slide, code,
                        left=left_position,
                        top=top_position,
                        width=content_width,
                        height=height,
                        font_size=10  # Smaller font for code
                    )
                    
                    # Format code textbox
                    text_frame = textbox.text_frame
                    for paragraph in text_frame.paragraphs:
                        paragraph.font.name = "Courier New"  # Monospace font
                    
                    top_position += height + 0.2
            
            elif item_type == 'paragraph':
                text = item.get('text', '')
                if text:
                    lines = len(text) // 100 + 1  # Rough estimate of number of lines
                    height = lines * 0.2 + 0.2
                    textbox = self.ppt.add_textbox(
                        slide, text,
                        left=left_position,
                        top=top_position,
                        width=content_width,
                        height=height
                    )
                    top_position += height + 0.2
            
            elif item_type == 'table':
                headers = item.get('headers', [])
                rows = item.get('rows', [])
                if headers and rows:
                    # Table implementation would be more complex
                    # This is a placeholder - would need to use shapes.add_table in a full implementation
                    # For now, we'll create a textual representation
                    table_text = "| " + " | ".join(headers) + " |\n"
                    table_text += "| " + " | ".join(["-"*len(h) for h in headers]) + " |\n"
                    for row in rows:
                        table_text += "| " + " | ".join(row) + " |\n"
                    
                    height = (len(rows) + 2) * 0.3
                    textbox = self.ppt.add_textbox(
                        slide, table_text,
                        left=left_position,
                        top=top_position,
                        width=content_width,
                        height=height,
                        font_size=10
                    )
                    
                    top_position += height + 0.2
    
    def _resolve_image_path(self, image_src: str, md_file_dir: Optional[Path] = None) -> str:
        """
        Resolve the path to an image file.
        
        Args:
            image_src: Source path or URL for the image
            md_file_dir: Directory of the source Markdown file
            
        Returns:
            Resolved local path to the image or empty string if not found
        """
        # Check if it's a URL
        if image_src.startswith(('http://', 'https://')):
            # We could download the image here, but for now just return empty
            self.logger.warning(f"URL images not currently supported: {image_src}")
            return ""
        
        # Try relative to markdown file directory
        if md_file_dir:
            path = md_file_dir / image_src
            if path.exists():
                return str(path)
        
        # Try relative to image directory if specified
        if self.image_dir:
            path = self.image_dir / image_src
            if path.exists():
                return str(path)
            
            # Try just the filename part
            image_filename = Path(image_src).name
            path = self.image_dir / image_filename
            if path.exists():
                return str(path)
        
        # Try as absolute path
        if Path(image_src).exists():
            return image_src
            
        self.logger.warning(f"Image not found: {image_src}")
        return ""
