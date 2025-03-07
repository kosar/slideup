import streamlit as st
import os
import sys
from pathlib import Path

# Guarantee the project root is in sys.path regardless of execution context
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can safely use absolute imports from the project root
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
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save uploaded file to a temporary location
            temp_input_path = os.path.join("/tmp", uploaded_file.name)
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            result = crew.enhance_presentation(temp_input_path, output_path, enhancement_options)
            st.success("Enhancement Successful!")
            st.write(f"Enhanced file saved to: {output_path}")
