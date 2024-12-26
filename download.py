import os
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
from subprocess import call  # Import to execute another Python file


parser = argparse.ArgumentParser()
parser.add_argument('--base_url', type=str, default='http://alab.ise.ous.ac.jp/robocupdata/', help='Base URL for downloading data')
parser.add_argument('--subpaths', type=str, nargs='+', required=True, help='List of subpaths to download data from')
parser.add_argument('--save_dir', type=str, default='robocup2d_data', help='Directory to save downloaded files')
parser.add_argument('--option', type=str, default=None)
args, _ = parser.parse_known_args()

# url = "http://alab.ise.ous.ac.jp/robocupdata/rc2021-roundrobin/normal/alice2021-helios2021/"
urls = [args.base_url + os.sep + subpath + os.sep for subpath in args.subpaths]
save_dir = args.save_dir
os.makedirs(args.save_dir, exist_ok=True)
debug = True

# Function to download data
def download_data(file_name):
    file_url = url + file_name
    file_path = os.path.join(save_dir, file_name)
    with requests.get(file_url, stream=True) as file_response:
        with open(file_path, 'wb') as file:
            for chunk in file_response.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Downloaded {file_name}")

# Downloading files
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    i = 0
    with ThreadPoolExecutor() as executor:
        for link in soup.find_all('a', href=True):
            if debug and i == 5:
                break
            file_name = link['href']
            if file_name.endswith("tracking.csv"):
                executor.submit(download_data, file_name)
                i += 1
