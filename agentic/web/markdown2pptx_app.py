import streamlit as st
import os
import sys
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Verify the import path
from agentic.crews.markdown2pptx_crew import MarkdownToPPTXCrew

def convert_markdown_to_pptx(markdown_text, template, api_keys):
    """Convert Markdown to PowerPoint using CrewAI."""
    crew = MarkdownToPPTXCrew(use_direct_calls=True)
    output_path = "/path/to/generated/presentation.pptx"
    result = crew.run(markdown_text, output_path)
    return result

st.title("Markdown to PowerPoint Converter")

markdown_input = st.text_area("Enter Markdown Text:", height=300)
uploaded_file = st.file_uploader("Upload Markdown File", type=["md"])

if uploaded_file is not None:
    markdown_input = uploaded_file.read().decode("utf-8")
    st.text_area("Markdown Content", markdown_input, height=300)

template = st.selectbox("Select PowerPoint Template", ["Default", "Template 1", "Template 2"])
openai_api_key = st.text_input("OpenAI API Key", type="password")
stability_api_key = st.text_input("Stability API Key", type="password")
deepseek_api_key = st.text_input("DeepSeek API Key", type="password")

if st.button("Convert to PowerPoint"):
    if markdown_input:
        with st.spinner("Converting..."):
            api_keys = {
                "openai": openai_api_key,
                "stability": stability_api_key,
                "deepseek": deepseek_api_key
            }
            output_path = convert_markdown_to_pptx(markdown_input, template, api_keys)
        st.success("Conversion Successful!")
        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download PowerPoint",
                data=file,
                file_name=os.path.basename(output_path),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
            )
    else:
        st.warning("Please enter some Markdown text.")
