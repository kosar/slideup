"""
Command-line interface for SlideUp.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from .md_to_pptx import MarkdownToPPTX
from .config import Config
from .utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="SlideUp - Convert Markdown to PowerPoint")
    
    # Main file input/output arguments
    parser.add_argument("input", nargs="?", help="Path to Markdown file or directory")
    parser.add_argument("-o", "--output", help="Path for the output PowerPoint file or directory")
    
    # Configuration options
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument("-t", "--template", help="Path to PowerPoint template file")
    parser.add_argument("-i", "--images", help="Directory containing images referenced in Markdown")
    
    # Presentation metadata
    parser.add_argument("--title", help="Presentation title (overrides title from Markdown)")
    parser.add_argument("--subtitle", help="Presentation subtitle")
    
    # Processing options
    parser.add_argument("-r", "--recursive", action="store_true", 
                      help="Process all Markdown files in directory recursively")
    parser.add_argument("--theme", help="Theme name to use for presentation")
    
    # Logging options
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output except for errors")
    
    # Utility commands
    parser.add_argument("--init", action="store_true", 
                      help="Initialize SlideUp configuration in current directory")
    parser.add_argument("--list-themes", action="store_true", help="List available presentation themes")
    parser.add_argument("--version", action="store_true", help="Show version information")
    
    return parser.parse_args()


def process_single_file(
    markdown_path: Path,
    output_path: Optional[Path],
    converter: MarkdownToPPTX,
    title: Optional[str] = None,
    subtitle: Optional[str] = None
) -> str:
    """
    Process a single Markdown file to PowerPoint.
    
    Args:
        markdown_path: Path to Markdown file
        output_path: Path to output PowerPoint file
        converter: MarkdownToPPTX converter instance
        title: Optional presentation title
        subtitle: Optional presentation subtitle
        
    Returns:
        Path to created PowerPoint file
    """
    return converter.convert_file(
        markdown_path=markdown_path,
        output_path=output_path,
        title=title,
        subtitle=subtitle
    )


def process_directory(
    input_dir: Path,
    output_dir: Path,
    converter: MarkdownToPPTX,
    recursive: bool = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None
) -> List[str]:
    """
    Process all Markdown files in a directory.
    
    Args:
        input_dir: Directory containing Markdown files
        output_dir: Directory for output PowerPoint files
        converter: MarkdownToPPTX converter instance
        recursive: Whether to process subdirectories recursively
        title: Optional presentation title
        subtitle: Optional presentation subtitle
        
    Returns:
        List of paths to created PowerPoint files
    """
    output_paths = []
    
    if recursive:
        markdown_files = list(input_dir.glob("**/*.md"))
    else:
        markdown_files = list(input_dir.glob("*.md"))
    
    for md_file in markdown_files:
        # Create relative path for output file
        if recursive:
            relative_path = md_file.relative_to(input_dir)
            output_subdir = output_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_file = output_subdir / (md_file.stem + ".pptx")
        else:
            output_file = output_dir / (md_file.stem + ".pptx")
        
        result = process_single_file(
            markdown_path=md_file,
            output_path=output_file,
            converter=converter,
            title=title,
            subtitle=subtitle
        )
        
        if result:
            output_paths.append(result)
    
    return output_paths


def show_version() -> None:
    """Show version information."""
    try:
        from importlib.metadata import version
        slideup_version = version("slideup")
    except (ImportError, ModuleNotFoundError):
        slideup_version = "unknown"
    
    print(f"SlideUp version: {slideup_version}")


def init_config(config_path: Path) -> None:
    """
    Initialize a new configuration file.
    
    Args:
        config_path: Path where to save the configuration
    """
    config = Config()
    config.save(config_path)
    print(f"Initialized SlideUp configuration at {config_path}")


def list_themes(config: Config) -> None:
    """
    List available presentation themes.
    
    Args:
        config: SlideUp configuration
    """
    themes = config.get("themes", {})
    if not themes:
        print("No themes available")
        return
    
    print("Available themes:")
    for theme_name in themes.keys():
        print(f"- {theme_name}")


def main() -> int:
    """
    Main entry point for the SlideUp CLI.
    
    Returns:
        Exit code
    """
    args = parse_args()
    
    # Setup logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    logger = setup_logger("slideup", level=log_level)
    
    # Handle utility commands
    if args.version:
        show_version()
        return 0
    
    # Load configuration
    config = Config(args.config)
    
    if args.init:
        init_config(Path("slideup.json"))
        return 0
    
    if args.list_themes:
        list_themes(config)
        return 0
    
    # Check for input file or directory
    if not args.input:
        logger.error("No input file or directory specified")
        return 1
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Determine template path
    template_path = args.template
    if not template_path:
        template_path = config.get("general.template_path")
    
    # Determine image directory
    image_dir = args.images
    if not image_dir:
        image_dir = config.get("general.image_dir")
    
    # Create converter
    converter = MarkdownToPPTX(
        template_path=template_path,
        image_dir=image_dir,
        output_dir=Path(config.get("general.output_dir")),
    )
    
    # Process input
    title = args.title
    subtitle = args.subtitle
    
    try:
        if input_path.is_file():
            # Process single file
            output_path = Path(args.output) if args.output else None
            result = process_single_file(
                markdown_path=input_path,
                output_path=output_path,
                converter=converter,
                title=title,
                subtitle=subtitle
            )
            
            if result:
                logger.info(f"Created presentation: {result}")
            else:
                logger.error("Failed to create presentation")
                return 1
                
        elif input_path.is_dir():
            # Process directory
            output_dir = Path(args.output) if args.output else Path(config.get("general.output_dir"))
            results = process_directory(
                input_dir=input_path,
                output_dir=output_dir,
                converter=converter,
                recursive=args.recursive,
                title=title,
                subtitle=subtitle
            )
            
            if results:
                logger.info(f"Created {len(results)} presentation(s)")
                if args.verbose:
                    for result in results:
                        logger.info(f"- {result}")
            else:
                logger.warning("No presentations were created")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import logging
    import traceback
    sys.exit(main())