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

# Only import SDV if the package is installed.
global use_sdv
if importlib.util.find_spec("sdv"):
    print("Importing SDV...", flush=True)
    import sdv
    use_sdv = True
    print("SDV module found.", flush=True)
else:
    use_sdv = False
    print("SDV module unavailable.", flush=True)

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


# One Market Model Feature Registry
#
# Centralized definitions of all modeling features,
# constraints, and priorities supported by AFCCP.
MODEL_FEATURES = {

    # ____________________________________ TIER 1 ____________________________________

    # AFSC Production
    "meet_afsc_pgl": {
        "name": "Meet AFSC PGL Targets",
        "priority": "Tier 1",
        "description": "Enforce AFSC production targets.",
        "default": False
    },

    # Rated Assignments
    "meet_rated_pgl": {
        "name": "Meet Rated PGL Targets",
        "priority": "Tier 1",
        "description": "Enforce rated production targets.",
        "default": False
    },

    "meet_rated_soc_pgl": {
            "name": "Meet Rated SOC PGL Targets",
            "priority": "Tier 2",  # Maybe this changes to just Tier 3 or something
            "description": "Enforce rated production targets by SOC.",
            "default": False
        },

    "rated_stability_soc": {
        "name": "Rated Stability by SOC",
        "priority": "Tier 1",
        "description": "Enforce rated matching stability by commissioning source using Algorithms 1/2. "
                       "These are the 'Matched/Reserves/Alternates' information gained from Algorithm 1/2.",
        "default": False
    },

    # ____________________________________ TIER 2 ____________________________________

    # Space Force
    "ussf_om_balance": {
        "name": "USSF Order of Merit Balance",
        "priority": "Tier 2",
        "description": "Enforce order of merit balancing for USSF.",
        "default": False,
        "threshold": 0.02  # Target is 50% +- {threshold}
    },

    # TODO: Implement this feature
    "ussf_degree_targets": {
        "name": "USSF Degree Targets",
        "priority": "Tier 2",
        "description": "Ensure USSF receives desired degree proportion breakouts.",
        "default": False
    },

    # USAFA Support AFSCs
    "usafa_support_cap": {
        "name": "USAFA Support AFSC Cap",
        "priority": "Tier 2",
        "description": "Limit USAFA cadets assigned to designated support AFSCs.",
        "default": False,
        "percent_cap": 0.05,
        "constrained_afscs": ['35P', '38F', '64P', '65F']

    },

    # Reward High OM
    "high_om_receive_preference": {
        "name": "High OM Receive Preferred AFSCs",
        "priority": "Tier 2",
        "description": "Ensure higher performing cadets (by OM) receive a preferred AFSC.",
        "default": False,
        "percentile_cutoff": 0.1,
        "preference_number": 3,
    },

    # CASTLE
    "castle_curves_utilized": {
        "name": "Utilize CASTLE Value Input",
        "priority": "Tier 2",
        "description": "Allocate surplus cadets intelligently based on needs of DAF using CASTLE.",
        "default": False
    }
}


def initialize_feature_checklist():
    return {
        feature: details.get("default", False)
        for feature, details in MODEL_FEATURES.items()
    }
