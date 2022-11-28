from pptx import Presentation
import afccp.core.globals


def generate_results_slides(instance):
    """
    Function to generate the results slides for a particular problem instance with solution
    """

    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]

    title.text = "Hello, World!"
    subtitle.text = "python-pptx was here!"

    # Save the PowerPoint
    filepath = paths_out['figures'] + instance.data_name + "/slides/" + instance.solution_name + ' Results.pptx'
    prs.save(filepath)