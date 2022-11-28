# Import libraries
print("Importing 'afccp' module...")  # I just like to see that something is happening...
import os
import pandas as pd
import openpyxl
from packaging import version

# Get directory path here
dir_path = os.getcwd() + '/'

# Check to see if we already have the data folders we need in our working directory
for folder in ["figures", "instances", "results", "solvers", "support"]:
    folder_exists = os.path.exists(folder)

    # If we don't have the folder, we make one
    if not folder_exists:
        os.makedirs(folder)

# Print update
if folder_exists:
    print("Data folders found.")
else:
    print("Data folders not found. Creating folders in working directory...")

# Folder paths!
global paths
paths = {}
for folder in ["figures", "instances", "results", "solvers", "support"]:
    paths[folder] = folder + '/'

# Only use pyomo script if we have pyomo
global use_pyomo
try:
    from pyomo.environ import *

    use_pyomo = True
    print('Pyomo module found.')
except:
    use_pyomo = False
    print('Pyomo module unavailable.')

# Only use sdv functions if we have sdv
global use_sdv
try:
    import sdv

    use_sdv = True
    print('SDV module found.')
except:
    use_sdv = False
    print('SDV module unavailable.')

# Only calculate similarity plots if we have sklearn manifold
global use_manifold
try:
    from sklearn import manifold

    use_manifold = True
    print('Sklearn Manifold module found.')
except:
    use_manifold = False
    print('Sklearn Manifold module unavailable.')


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
