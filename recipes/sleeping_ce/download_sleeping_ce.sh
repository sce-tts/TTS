# create venv
python -m venv env
source env/Scripts/activate
pip install pip --upgrade

# download dataset
pip install gdown
gdown --id 1UpoBaZRTJXkTdsoemLBWV48QClm6hpTX -O filelists.zip
unzip -q filelists.zip -d ./sleeping-ce

rm filelists.zip
rm -rf env
