# Import libraries
import os
import pandas as pd
import openpyxl
from packaging import version

# Get directory path
global dir_path
dir_path = os.getcwd() + '/'

# Check if the directory is actually in the afccp sub-folder, not the main folder
if dir_path.count('afccp') != 1:
    dir_path = dir_path[:-6]

# Additional folders in Directory
global paths
resource_path = 'afccp/resources/'
paths = {}
for folder in ['figures', 'instances', 'results', 'solvers', 'support', 'tables']:
    paths[folder] = dir_path + resource_path + folder + '/'

# sensitive information
global sensitive_folder
sensitive_folder = os.path.exists(dir_path + 'afccp/sensitive')

if sensitive_folder:
    print('Sensitive data folder found.')
else:
    print('Sensitive data folder not found.')

# Additional sensitive folders
sensitive_path = 'afccp/sensitive/resources/'
for folder in ['instances', 'results', 'support', 'raw']:
    paths['s_' + folder] = dir_path + sensitive_path + folder + '/'

# This determines how we import data from excel!
global specify_engine
if version.parse(pd.__version__) > version.parse("1.2.1"):
    specify_engine = True
else:
    specify_engine = False

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
    import .xlsx files normally but some have to add ", engine= 'openpyxl'". Pandas versions > 1.2.1 must
    specify openpyxl as the engine
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
