# Import libraries
import os
import pandas as pd
import openpyxl
from packaging import version

# Get directory path
global dir_path, exe_extension, databricks, paths_in, paths_out, support_paths, provide_executable, executable
dir_path = os.getcwd() + '/'
exe_extension = True  # specific variable relating to pyomo solver paths
databricks = False  # initially assume we're not running on databricks
provide_executable, executable = True, None  # Global variables to determine how to work with pyomo

# This determines how we import data from excel!
global specify_engine
if version.parse(pd.__version__) > version.parse("1.2.1"):
    specify_engine = True
else:
    # specify_engine = False  # Looks like we need to specify engine all the time! (databricks issue)
    specify_engine = True

# Figure out where this directory is running from
if 'databricks' in dir_path:  # '/databricks/driver/' is the databricks working directory

    # Databricks
    print('Running on databricks.')
    databricks = True

    # Decided to use databricks in a more generalized fashion
    input_folder = dir_path + 'afccp/resources/shared/'
    output_folder = dir_path + 'afccp/resources/shared/'
    support_folder = dir_path + 'afccp/resources/shared/'

    # Pyomo global variables
    provide_executable = False

elif 'workspace' in dir_path:  # We're running this from plotly!

    # We're on my macbook
    print("Running on plotly enterprise")

    # Import and export to my folder
    input_folder = dir_path + 'afccp/resources/laird/'
    output_folder = dir_path + 'afccp/resources/laird/'
    support_folder = dir_path + 'afccp/resources/shared/'

    # Pyomo global variables
    provide_executable = False

elif 'griffenlaird' in dir_path:

    # We're on my macbook
    print("Running on Griffen's Macbook")

    # If I'm on my Mac, I don't want to add ".exe" to the solver path
    exe_extension = False

    # # Pyomo global variables
    # provide_executable = False

    # Import and export to my folder
    input_folder = dir_path + 'afccp/resources/laird/'
    output_folder = dir_path + 'afccp/resources/laird/'
    support_folder = dir_path + 'afccp/resources/shared/'

elif 'griff007' in dir_path:

    # We're on my old macbook
    print("Running on Griffen's old macbook")

    # Import and export to my folder
    input_folder = dir_path + 'afccp/resources/laird/'
    output_folder = dir_path + 'afccp/resources/laird/'
    support_folder = dir_path + 'afccp/resources/shared/'

elif 'ianmacdonald' in dir_path:

    # We're on my old macbook
    print("Running on Ian's macbook")

    # Import and export to my folder
    input_folder = dir_path + 'afccp/resources/macdonald/'
    output_folder = dir_path + 'afccp/resources/macdonald/'
    support_folder = dir_path + 'afccp/resources/shared/'

    # Turn this back to True if you use ", engine='openpyxl'" in your pd.read_excel() statements
    specify_engine = False

else:

    # Running somewhere else
    print("Running elsewhere")

    # Import and export to shared folder
    input_folder = dir_path + 'afccp/resources/shared/'
    output_folder = dir_path + 'afccp/resources/shared/'
    support_folder = dir_path + 'afccp/resources/shared/'


# Need different sets of paths (data to import from and to export to)
paths_in = {}  # Path to the folders we want to import data from
paths_out = {}  # Path to the folders we want to export data to
support_paths = {}  # Path to the folders we want to import supporting material from (and potentially export to)

# Paths "in"
paths_in = {}
for folder in ['figures', 'instances', 'results', 'tables']:
    paths_in[folder] = input_folder + folder + '/'

# Paths "out"
paths_out = {}
for folder in ['figures', 'instances', 'results', 'tables']:

    if output_folder == "":
        paths_out[folder] = ""  # Just want to export to the working directory
    else:
        paths_out[folder] = output_folder + folder + '/'

# Support paths
for folder in ['real', 'scrubbed', 'solvers']:
    support_paths[folder] = support_folder + folder + '/'

# sensitive information
global sensitive_folder, sensitive_folder
sensitive_folder = os.path.exists(dir_path + 'afccp/sensitive')

if sensitive_folder:
    print('Sensitive data folder found.')
else:
    print('Sensitive data folder not found.')

# Additional sensitive folder path (for the original thesis data cleaning)
sensitive_path = dir_path + 'afccp/sensitive/raw/'

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
