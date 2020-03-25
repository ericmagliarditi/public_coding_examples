# This software program enables users to download multiple images from the Planet database. 
  
# File Structure
    .
    ├── planet_data_download  # This Repository
    │   ├── bin               # Directory with Python files to execute training
    │   ├──── download_images.py   #Executable Python script
    │   ├── planetpy          # Directory with Python files that contain all backend utilities, etc. 
    │   ├──── __init__.py          #Initialization Python script
    │   ├──── util.py              #Contains all necessary utility functions and classes
    │   ├── setup.py          # Prepares your system to run all source code - installs dependencies and connects all modules
    │   ├── README.me           
    │   └── ...                
    └── ...

# Command Line Arguments:
    --start-year: Start year for analysis
    --start-month: Start month for analysis
    --start-day: Start day for analysis
    --cloud-percent: Filter out images that are more than X percent cloud covered
    --geo-json-path: File path for the geo json document [ROI coordinates] - REQUIRED ARGUMENT
    --item-types: Satellite item types that are available - List type required. Example: ['PSScene3Band', 'PSScene4Band']
    --asset-type: Represents the type of asset such as visual or analytic from the satellite - List type required. Example: ['visual']
    --output-dir: Directory where assets are saved - REQUIRED ARGUMENT
    

# 1. Setup all dependencies:
    pip install -e .
  
# 2. Example Command Line Argument
    python bin/download_images.py --start-year 2009 --start-month 1 --start-day 1 --geo-json-path examples/seaport_geometry.json --output-dir ../../../Desktop/test/ --cloud-percent 0.5




    

    
    
    
    
 
 

    
