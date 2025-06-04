from afccp.data.generation.basic import *

# Import 'realistic' generation functions if we have the SDV module
from afccp.globals import use_sdv
if use_sdv:
    from afccp.data.generation.realistic import *