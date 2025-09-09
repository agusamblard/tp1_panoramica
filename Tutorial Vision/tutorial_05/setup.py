import os
from i308_utils import download, unzip

def download_calib_dataset():

    download(
        "https://github.com/udesa-vision/i308-resources/raw/refs/heads/main/tutoriales/tutorial_05/images/images.zip"
    )
    unzip(
        "images.zip",
    )

    os.remove("images.zip")
