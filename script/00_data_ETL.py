# data downloaded to zip in different directory
import zipfile

# get location of Download zip relative to notbebook 
## mine is outside source control here ../../HumanaProject/_data/Download.zip
zip_dir = input("Where's the goods (Download.zip)?:")

# define data directory

# extract data to project folder
with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
    zip_ref.extractall(zip_dir)