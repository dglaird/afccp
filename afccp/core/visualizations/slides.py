from pptx import Presentation
from pptx.util import Inches
import afccp.core.globals
import os
import copy
import numpy as np
import pandas as pd


def generate_results_slides(instance):
    """
    Function to generate the results slides for a particular problem instance with solution
    """

    # Build the presentation object
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    blank_slide_layout = prs.slide_layouts[6]

    # Add a slide
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]

    # Add some text
    title.text = "Hello, World!"
    subtitle.text = "python-pptx was here!"

    # Size of the chart
    top, left = Inches(instance.mdl_p['ch_top']), Inches(instance.mdl_p['ch_left'])
    height, width = Inches(instance.mdl_p['ch_height']), Inches(instance.mdl_p['ch_width'])

    # Get the file paths to all the relevant images
    folder_path = instance.export_paths['Analysis & Results'] + "Results Charts/"
    folder = os.listdir(folder_path)
    chart_paths = {}
    for file in folder:
        if instance.solution_name in file:
            chart_paths[file] = folder_path + file

    # Loop through each image file path to add the image to the presentation
    for file in chart_paths:

        # Add an empty slide
        slide = prs.slides.add_slide(blank_slide_layout)

        # Add the picture to the slide
        slide.shapes.add_picture(chart_paths[file], left, top, height=height, width=width)

    # Save the PowerPoint
    filepath = instance.export_paths['Analysis & Results'] + instance.data_name + ' ' + instance.solution_name + '.pptx'
    prs.save(filepath)

def create_animation_slides(instance):
    """
    Function to generate the results slides for a particular problem instance with solution
    """

    # Shorthand
    mdl_p, sequence = instance.mdl_p, instance.solution_iterations['sequence']

    # Build the presentation object
    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]

    # Set width and height of presentation
    prs.slide_width = Inches(mdl_p['b_figsize'][0])
    prs.slide_height = Inches(mdl_p['b_figsize'][1])

    # Size of the chart
    top, left = Inches(0), Inches(0)
    height, width = Inches(mdl_p['b_figsize'][1]), Inches(mdl_p['b_figsize'][0])

    # Make sure we have the "sequence" folder
    if sequence not in os.listdir(instance.export_paths['Analysis & Results'] + "Cadet Board/"):
        raise ValueError("Error. Sequence folder '" + sequence + "' not in 'Cadet Board' folder.")

    # Get the file path to the sequence folder
    sequence_folder_path = instance.export_paths['Analysis & Results'] + "Cadet Board/" + sequence + '/'

    # Make sure we have the "focus" folder
    if mdl_p['focus'] not in os.listdir(sequence_folder_path):
        raise ValueError("Error. Focus folder '" + mdl_p['focus'] + "' not in '" + sequence + "' sequence folder.")

    # Files in the focus folder
    focus_folder_path = sequence_folder_path + mdl_p['focus'] + '/'
    folder = np.array(os.listdir(focus_folder_path))

    # Regular solution iterations frames: 1.png for example
    if "1.png" in folder:

        # Sort the folder in order by the frames
        int_vals = np.array([int(file[0]) for file in folder])
        indices = np.argsort(int_vals)
        img_paths = {file: focus_folder_path + file for file in folder[indices]}

    # Matching Algorithm proposals/rejections frames: 1 (Proposals).png & 1 (Rejections).png for example
    else:

        # The integer values at the beginning of each file
        int_vals = np.array([int(file[:2]) for file in folder])
        min, max = np.min(int_vals), np.max(int_vals)

        # Loop through the files to get the ordered list of frames
        img_paths = {}
        for val in np.arange(min, max + 1):
            indices = np.where(int_vals == val)[0]
            files = folder[indices]
            if len(files) > 1:
                if 'Proposals' in files[0]:
                    img_paths[files[0]] = focus_folder_path + files[0]
                    img_paths[files[1]] = focus_folder_path + files[1]
                else:
                    img_paths[files[1]] = focus_folder_path + files[1]
                    img_paths[files[0]] = focus_folder_path + files[0]
            else:
                img_paths[files[0]] = focus_folder_path + files[0]

    # Loop through each image file path to add the image to the presentation
    for file in img_paths:

        # Add an empty slide
        slide = prs.slides.add_slide(blank_slide_layout)

        # Add the picture to the slide
        slide.shapes.add_picture(img_paths[file], left, top, height=height, width=width)

    # Save the PowerPoint
    filepath = sequence_folder_path + mdl_p['focus'] + '.pptx'
    prs.save(filepath)