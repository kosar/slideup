import os
import sys
import tempfile
import streamlit as st
from pathlib import Path
import base64
from io import BytesIO

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import crews
from agentic.crews.markdown2pptx_crew import Markdown2PPTXCrew
from agentic.crews.pptx_enhancement_crew import PPTXEnhancementCrew

# Import utilities
from agentic.utils.file_utils import save_uploaded_file, create_download_link

# Page configuration
st.set_page_config(
    page_title="SlideUp AI - Presentation Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .css-1544g2n {
        padding: 2rem 1rem;
    }
    .upload-box {
        border: 2px dashed #6c757d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.title("SlideUp AI - Presentation Assistant")
    st.markdown("### Transform your content into professional presentations")
    
    # Navigation using tabs
    tab1, tab2 = st.tabs(["Markdown to PowerPoint", "PowerPoint Enhancement"])
    
    # Tab 1: Markdown to PowerPoint
    with tab1:
        markdown_to_powerpoint_app()
    
    # Tab 2: PowerPoint Enhancement
    with tab2:
        powerpoint_enhancement_app()

def create_temp_directory():
    """Create a temporary directory for file operations"""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def display_file_download_button(file_path, button_text, mime_type="application/octet-stream"):
    """Display a download button for the specified file"""
    with open(file_path, "rb") as file:
        file_bytes = file.read()
        
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{os.path.basename(file_path)}" class="download-button">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def markdown_to_powerpoint_app():
    """Implement the Markdown to PowerPoint functionality"""
    st.header("Convert Markdown to PowerPoint")
    
    with st.expander("ℹ️ How to use this tool", expanded=False):
        st.markdown("""
        **Instructions:**
        1. Upload a Markdown (.md) file or paste your Markdown content in the text area
        2. Configure the presentation settings
        3. Click 'Generate PowerPoint' to create your presentation
        4. Download the generated PowerPoint file
        
        **Tips for good results:**
        - Use proper Markdown formatting with headers (# for titles, ## for sections, etc.)
        - Include bullet points and numbered lists for better slide organization
        - Keep paragraphs concise for better slide layout
        """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Input")
        input_method = st.radio("Select input method:", ["Upload Markdown File", "Paste Markdown Content"])
        
        markdown_content = None
        markdown_file = None
        
        if input_method == "Upload Markdown File":
            uploaded_file = st.file_uploader("Choose a Markdown file", type=["md"])
            if uploaded_file is not None:
                markdown_file = uploaded_file
                markdown_content = uploaded_file.getvalue().decode("utf-8")
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                with st.expander("Preview Content", expanded=False):
                    st.markdown(markdown_content)
        else:
            markdown_content = st.text_area("Enter your Markdown content:", height=300)
    
    with col2:
        st.subheader("Configuration")
        theme = st.selectbox("Presentation Theme:", ["Professional", "Creative", "Minimalist", "Academic", "Corporate"])
        color_scheme = st.selectbox("Color Scheme:", ["Blue", "Green", "Red", "Purple", "Orange", "Grayscale"])
        include_cover = st.checkbox("Include cover slide", value=True)
        include_agenda = st.checkbox("Include agenda slide", value=True)
        
    if st.button("Generate PowerPoint", type="primary", disabled=not markdown_content):
        if markdown_content:
            with st.spinner("Generating your PowerPoint presentation..."):
                try:
                    # Create a temporary directory for processing
                    temp_dir = create_temp_directory()
                    
                    # Save the markdown content to a temp file if it was pasted
                    if markdown_file is None:
                        markdown_path = os.path.join(temp_dir, "input.md")
                        with open(markdown_path, "w") as f:
                            f.write(markdown_content)
                    else:
                        markdown_path = os.path.join(temp_dir, markdown_file.name)
                        save_uploaded_file(markdown_file, markdown_path)
                    
                    # Set output path
                    output_file = os.path.join(temp_dir, "generated_presentation.pptx")
                    
                    # Initialize the crew and run the conversion
                    crew = Markdown2PPTXCrew()
                    result = crew.run(
                        markdown_file=markdown_path,
                        output_file=output_file,
                        theme=theme,
                        color_scheme=color_scheme,
                        include_cover=include_cover,
                        include_agenda=include_agenda
                    )
                    
                    # Display success message and download button
                    st.success("PowerPoint presentation generated successfully!")
                    display_file_download_button(
                        output_file, 
                        "💾 Download PowerPoint Presentation", 
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check your Markdown content and try again.")

def powerpoint_enhancement_app():
    """Implement the PowerPoint Enhancement functionality"""
    st.header("Enhance Your PowerPoint Presentation")
    
    with st.expander("ℹ️ How to use this tool", expanded=False):
        st.markdown("""
        **Instructions:**
        1. Upload your PowerPoint (.pptx) file
        2. Select the enhancement options you want to apply
        3. Click 'Enhance PowerPoint' to improve your presentation
        4. Download the enhanced presentation
        
        **Available enhancements:**
        - Visual design improvements
        - Content refinement
        - Structure optimization
        - Slide formatting
        """)
    
    uploaded_file = st.file_uploader("Upload your PowerPoint presentation", type=["pptx"])
    
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        st.subheader("Enhancement Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            improve_visuals = st.checkbox("Improve visual design", value=True)
            enhance_content = st.checkbox("Enhance content quality", value=True)
        
        with col2:
            optimize_structure = st.checkbox("Optimize presentation structure", value=True)
            improve_formatting = st.checkbox("Improve slide formatting", value=True)
        
        enhancement_level = st.select_slider(
            "Enhancement Level",
            options=["Light", "Moderate", "Comprehensive"],
            value="Moderate"
        )
        
        additional_notes = st.text_area(
            "Any specific instructions or focus areas?",
            placeholder="E.g., 'Make it more suitable for executive audience' or 'Focus on simplifying complex concepts'"
        )
        
        if st.button("Enhance PowerPoint", type="primary"):
            with st.spinner("Enhancing your presentation... This may take a few minutes..."):
                try:
                    # Create a temporary directory for processing
                    temp_dir = create_temp_directory()
                    
                    # Save the uploaded file
                    input_pptx_path = os.path.join(temp_dir, uploaded_file.name)
                    save_uploaded_file(uploaded_file, input_pptx_path)
                    
                    # Set output path
                    output_file = os.path.join(temp_dir, f"enhanced_{uploaded_file.name}")
                    
                    # Initialize the crew and run the enhancement
                    crew = PPTXEnhancementCrew()
                    result = crew.run(
                        input_file=input_pptx_path,
                        output_file=output_file,
                        improve_visuals=improve_visuals,
                        enhance_content=enhance_content,
                        optimize_structure=optimize_structure,
                        improve_formatting=improve_formatting,
                        enhancement_level=enhancement_level.lower(),
                        additional_instructions=additional_notes
                    )
                    
                    # Display success message and download button
                    st.success("PowerPoint presentation enhanced successfully!")
                    display_file_download_button(
                        output_file, 
                        "💾 Download Enhanced PowerPoint", 
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error("Please check your PowerPoint file and try again.")

# Run the app
if __name__ == "__main__":
    main()
