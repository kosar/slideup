I'll continue updating the code with the DeepSeek API integration and enhanced fact-based content for the podcast-to-video converter.

Let me explain the key enhancements I've made to the podcast-to-video converter:

## 1. DeepSeek API Integration

The code now intelligently detects and uses the DeepSeek API when available:

```python
# Use DeepSeek if available, otherwise fallback to OpenAI
if deepseek_api_key:
    print("Using DeepSeek API for language models")
    ai_client = OpenAI(
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com/v1"
    )
    chat_model = "deepseek-v3"
    embedding_model = "deepseek-v3-embedding"
else:
    print("Using OpenAI API for language models")
    ai_client = OpenAI(api_key=openai_api_key)
    chat_model = "gpt-4o"
    embedding_model = "text-embedding-3-large"
```

This setup automatically uses DeepSeek v3 for all AI operations when the key is present, with appropriate fallbacks to OpenAI when necessary.

## 2. Fact-Based Content Enhancement

I've completely redesigned the content enhancement workflow to focus on factually accurate, research-based information:

1. **Precise Fact Extraction**: The system now carefully extracts only the explicitly mentioned facts, names, concepts, and terminology from the transcript.

2. **Research-Based Enhancement**: For each extracted element, it conducts thorough research to ensure historical accuracy.

3. **Triple-Layer Spell Checking**:
   - Initial content generation with strict spelling requirements
   - Dedicated spell-checking pass for all textual content
   - Final spell-check for all image prompts

## 3. Subtitle Generation and Integration

The code now automatically creates and integrates subtitles:

1. **Multiple Subtitle Formats**: Generates both SRT and WebVTT formats for maximum compatibility.

2. **Flexible Display Options**: Allows control of subtitle display with three modes:
   - `always`: Show subtitles throughout the video
   - `important_parts`: Show subtitles only for key segments
   - `none`: Hide subtitles entirely

3. **Dual Output Options**: Creates two video files:
   - Standard version with external subtitle files
   - Hardcoded version with embedded subtitles for platforms like TikTok

## 4. Enhanced Visual Quality

The image generation has been significantly improved:

1. **Historical Accuracy Prompting**: All image prompts are enhanced with instructions to prioritize historical accuracy and educational value.

2. **Custom Image Parameters**: Using higher CFG scale values (8.0) to ensure the generated images more closely follow the prompts.

3. **Intelligent Positioning**: Better layout of visuals and text to avoid overlapping with subtitles.

## Usage Examples

```bash
# Basic usage with all defaults (always show subtitles)
python podcast_enhancer.py --audio podcast.mp3 --output enhanced_video.mp4

# Hide subtitles completely
python podcast_enhancer.py --audio podcast.mp3 --subtitle none

# Only show subtitles for important segments
python podcast_enhancer.py --audio podcast.mp3 --subtitle important_parts

# Skip transcription if you already have subtitle files
python podcast_enhancer.py --audio podcast.mp3 --skip_transcription
```

The enhanced code now delivers a superior podcast-to-video conversion that places fact-based, educational content at its core, while maintaining high production values with proper spelling, accurate visuals, and flexible subtitle options.