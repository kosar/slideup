import streamlit as st
import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Verify the import path
from agentic.crews.pptx_enhancement_crew import PPTXEnhancementCrew

st.title("PowerPoint Enhancement App")

uploaded_file = st.file_uploader("Upload PowerPoint Presentation", type=["pptx"])

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")

    selected_options = st.multiselect(
        "Select Enhancement Options",
        ["Improve Design", "Add Animations", "Optimize Content"]
    )

    # Convert selected options to a dictionary
    enhancement_options = {
        "improve_design": "Improve Design" in selected_options,
        "add_animations": "Add Animations" in selected_options,
        "optimize_content": "Optimize Content" in selected_options
    }

    if selected_options:
        st.write(f"Selected enhancements: {', '.join(selected_options)}")

    if st.button("Enhance Presentation"):
        with st.spinner("Enhancing..."):
            crew = PPTXEnhancementCrew()
            output_path = "./output/enhanced_presentation.pptx"
            
            # Save uploaded file to a temporary location
            temp_input_path = os.path.join("/tmp", uploaded_file.name)
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            result = crew.enhance_presentation(temp_input_path, output_path, enhancement_options)
            st.success("Enhancement Successful!")
            st.write(f"Enhanced file saved to: {output_path}")
