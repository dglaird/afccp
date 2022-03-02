# ____INPUT DIRECTORY PATH____
global dir_path
dir_path = "C:/Users/Griffen Laird/Desktop/AFIT/THESIS/Main Directory/"

# Additional folders in Directory
global paths
paths = {}
for folder in ['Analysis & Results', 'Charts & Figures', 'Data Cleaning', 'Data Processing Support',
               'Problem Instances', 'Tables', 'Solvers']:
    paths[folder] = dir_path + folder + '/'

# This determines how we import data from excel!
global specify_engine
specify_engine = True  # ____INPUT HERE____

# Import libraries
import pandas as pd
if specify_engine:
    import openpyxl

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
def import_data(filepath, sheet_name=None):
    """
    This function is to alleviate issues with importing pandas dataframes since some versions can just
    import .xlsx files normally but some have to add ", engine= 'openpyxl'"
    :param filepath: excel file path
    :param sheet_name: name of the sheet to import
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
