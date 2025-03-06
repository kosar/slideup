import os
from pptx import Presentation
from pptx.util import Inches, Pt

class PPTXEnhancer:
    def __init__(self, input_file, recommendations):
        # Load the existing PPTX and store recommendations.
        self.input_file = input_file
        self.recommendations = recommendations  # e.g. { "design": "Improve margins and alignment", "text": "Simplify text and improve clarity", "visual": "Add consistent imagery and icons" }
        self.prs = Presentation(input_file)

    def improve_slide_design(self, slide):
        # Enhance slide design and layout based on recommendations.
        # For example, if design recommendation exists, shift shapes to a left margin of 0.5 inches.
        design_rec = self.recommendations.get("design", "")
        if "margins" in design_rec.lower():
            for shape in slide.shapes:
                if hasattr(shape, 'left'):
                    # Adjust left position to a fixed margin if not already close
                    if shape.left > Inches(0.5):
                        shape.left = Inches(0.5)
        for shape in slide.shapes:
            if hasattr(shape, 'text'):
                # Example: adjust left margin if needed.
                pass  # Placeholder for improved layout logic.
        return slide

    def enhance_text(self, slide):
        # Improve text clarity and conciseness.
        text_rec = self.recommendations.get("text", "")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    # If text is long, reduce font size for better clarity.
                    if len(paragraph.text) > 100 and "simplify" in text_rec.lower():
                        paragraph.font.size = Pt(16)
                    else:
                        paragraph.font.size = Pt(18)
        if hasattr(slide, "shapes"):
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        # Example: set a unified font size.
                        paragraph.font.size = Pt(18)
        return slide

    def add_visual_elements(self, slide):
        # Add missing visual elements or improve existing ones.
        visual_rec = self.recommendations.get("visual", "")
        # Only add a placeholder if no picture exists in slide.
        pict_found = any(hasattr(shape, 'image') for shape in slide.shapes)
        if not pict_found and "imagery" in visual_rec.lower():
            placeholder_path = os.path.join(os.path.dirname(__file__), 'placeholder.png')
            try:
                # Position the placeholder at a fixed location.
                slide.shapes.add_picture(placeholder_path, Inches(8), Inches(1), width=Inches(2))
            except Exception as e:
                print(f"Error adding visual element: {e}")
        placeholder_path = os.path.join(os.path.dirname(__file__), 'placeholder.png')
        try:
            # For demonstration add a placeholder image.
            slide.shapes.add_picture(placeholder_path, Inches(1), Inches(1), width=Inches(2))
        except Exception as e:
            print(f"Error adding visual element: {e}")
        return slide

    def ensure_consistency(self):
        # Enforce consistency across slides (e.g. uniform title styles).
        for slide in self.prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame and 'title' in shape.name.lower():
                    for paragraph in shape.text_frame.paragraphs:
                        paragraph.font.name = "Calibri"
                        paragraph.font.bold = True
                        paragraph.font.size = Pt(28)
        for slide in self.prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame and 'title' in shape.name.lower():
                    for paragraph in shape.text_frame.paragraphs:
                        paragraph.font.size = Pt(28)
        return self.prs

    def enhance(self, output_file):
        # Apply all enhancement steps to each slide.
        for slide in self.prs.slides:
            slide = self.improve_slide_design(slide)
            slide = self.enhance_text(slide)
            slide = self.add_visual_elements(slide)
        self.ensure_consistency()
        self.prs.save(output_file)
        print(f"Enhanced PPTX saved as {output_file}")

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pptx_enhancer.py <input_pptx> <output_pptx>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # Dummy recommendations; in practice these come from user input or a CrewAI workflow.
    recommendations = {
        "design": "Improve margins and alignment",
        "text": "Simplify text and improve clarity",
        "visual": "Add consistent imagery and icons"
    }
    enhancer = PPTXEnhancer(input_file, recommendations)
    enhancer.enhance(output_file)

if __name__ == '__main__':
    main()
