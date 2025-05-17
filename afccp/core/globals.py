import os
import pandas as pd
import openpyxl
from packaging import version
import importlib.util

# Import libraries
print("Importing 'afccp' module...")  # I just like to see that something is happening...
dir_path = os.getcwd() + "/"

# Folders & Paths
global paths
paths = {"instances": dir_path + "instances/",
         "solvers": dir_path + "solvers/",
         "support": dir_path + "support/",
         "files": dir_path + "files/"}
for folder in paths:

    # If we don't have the folder, we make one
    if not os.path.exists(folder):
        print("Folder '" + folder + "' not in current working directory. Creating it now...")
        os.makedirs(folder)

    # Necessary support sub-folders
    if folder == 'support':
        for sub_folder in ['data', 'value parameters defaults']:
            sub_folder_path = paths['support'] + sub_folder

            # If we don't have the folder, we make one
            if not os.path.exists(sub_folder_path):
                print("Support sub-folder '" + sub_folder + "' not in current working directory. Creating it now...")
                os.makedirs(sub_folder_path)

# Available instances
global instances_available
instances_available = []
for data_name in os.listdir(paths["instances"]):
    if os.path.isdir(paths["instances"] + data_name):
        instances_available.append(data_name)

# Only import pyomo if we have the package installed!
global use_pyomo
if spec := importlib.util.find_spec("pyomo"):
    from pyomo.environ import *
    use_pyomo = True
    print("Pyomo module found.")
else:
    use_pyomo = False
    print("Pyomo module unavailable.")

# Only import sdv if we have the package installed!
global use_sdv
if spec := importlib.util.find_spec("sdv"):
    import sdv
    use_sdv = True
    print("SDV module found.")
else:
    use_sdv = False
    print("SDV module unavailable.")

# Only import sklearn.manifold if we have the package installed!
global use_manifold
if spec := importlib.util.find_spec("sklearn.manifold"):
    from sklearn import manifold
    use_manifold = True
    print("Sklearn Manifold module found.")
else:
    use_manifold = False
    print("Sklearn Manifold module unavailable.")

# Only import pptx if we have the package installed!
global use_pptx
if spec := importlib.util.find_spec("pptx"):
    from pptx import Presentation
    use_pptx = True
    print("Python PPTX module found.")
else:
    use_pptx = False
    print("Python PPTX module unavailable.")

# AFSC Objective Label Dictionary
global obj_label_dict
obj_label_dict = {"Merit": "Average Merit", "USAFA Proportion": "USAFA Proportion",
                  "Combined Quota": "Number of Cadets", "USAFA Quota": "Number of USAFA Cadets",
                   "ROTC Quota": "Number of ROTC Cadets", "OTS Quota": "Number of OTS Cadets",
                  "Mandatory": "Mandatory Degree Tier Proportion",
                   "Desired": "Desired Degree Tier Proportion", "Permitted":
                   "Permitted Degree Tier Proportion", "Male": "Proportion of Male Cadets",
                   "Minority": "Proportion of Non-Caucasian Cadets", "Utility": "Average Cadet Utility",
                   "Norm Score": "Normalized Preference Score", "Tier 1": "Degree Tier 1 Proportion",
                   "Tier 2": "Degree Tier 2 Proportion", "Tier 3": "Degree Tier 3 Proportion",
                   "Tier 4": "Degree Tier 4 Proportion"}


# Importing pandas dataframe function
def import_data(filepath, sheet_name=None, specify_engine=True):
    """
    This function is to alleviate issues with importing pandas dataframes since some versions can just
    import .xlsx files normally but some have to add ", engine= 'openpyxl'". Pandas versions > 1.2.1 must
    specify openpyxl as the engine
    :param filepath: excel file path
    :param sheet_name: name of the sheet to import
    :param specify_engine: issues with pandas "engine="
    :return: pandas dataframe
    """

    if specify_engine:
        if sheet_name is None:
            df = pd.read_excel(filepath, engine='openpyxl')
        else:
            df = pd.read_excel(filepath, sheet_name=sheet_name, engine='openpyxl')
    else:
        if sheet_name is None:
            df = pd.read_excel(filepath)
        else:
            df = pd.read_excel(filepath, sheet_name=sheet_name)

    return df

def import_csv_data(filepath):
    """
    My own import statement in case I change the way I import data later
    """
    return pd.read_csv(filepath)  #, encoding='latin-1')
