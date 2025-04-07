import gdown
import os
import argparse
import zipfile

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--subpaths', type=str, nargs='+', required=True, help='List of subpaths (zip file names without .zip extension) to download data for')
parser.add_argument('--save_dir', type=str, default='downloaded_files', help='Directory to save downloaded files')
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

subpath_to_file_id = {
    "yushan2024-robocin2024": "1e_PoJA5424xDnY6LKBan_7qhmEW7TvtU",
    "yushan2024-r2d2": "1pUFMngH05mNa4Hnd4l32g6FAylqwFGQu",
    "yushan2024-oxsy2024": "1A8mnHLpDZAjqwWFoLuvlQZvBsmW_Czlf",
    "yushan2024-mars2024": "1-4jzAfowitTGedKgHH5QgPd9nAM6y5qS",
    "yushan2024-itandroids2024": "1mFjf9bpMfEOosQ615SP_8yG0E7PLeOTG",
    "yushan2024-helios2024": "1V8oYJ5aBsVeiiTf2lCnPeMmnj45qxvY1",
    "yushan2024-helios2023": "1wHebTe_ZqK7qzzQ0rXzZ7mdQmQIqW6SZ",
    "yushan2024-helios2022": "1N9TsINVaEVs_wk0hcn8HSSv0mIDxj-6V",
    "yushan2024-fra2024": "1UCqEAynnckjseDu-R67B-CTSGeCK02MS",
    "yushan2024-cyrus2024": "199RDsq-Yi4SWZc6eD327C8poYpr15iyy",
    "yushan2024-aeteam2024": "1Abv8vVtrAbWlGE3Gf6Lb4ej5DtcLW6GL",
    "robocin2024-yushan2024": "1GCQCzbB8PUxwgbq08JMCgcvYSntPOgKv",
    "robocin2024-r2d2": "15Byri4irhFQgz6xMAgt5OlFK1jLlBZrw",
    "robocin2024-oxsy2024": "16pQl0lqmfU9MXVWPLKU1endVKyPT1rxt",
    "robocin2024-mars2024": "14oXPHMLnsR9Oc1Qx_Hwg-HwOIj99goL_",
    "robocin2024-itandroids2024": "1TTAfaQZUXeBQkg_EIQgt8n9S9_e9zb3l",
    "robocin2024-helios2024": "1DLELMq1qGP6RsOOnCn9rKTU90fH0D0jQ",
    "robocin2024-helios2023": "18ysLMxn0UnSfYHTepYA-fgIQe8l3aHRX",
    "robocin2024-helios2022": "1tIvYRC3knMIoctuaoMJSm1arNuJZqvwu",
    "robocin2024-fra2024": "1q6jsyGVabhu_mEfIa6HIKsYzahKD_2KV",
    "robocin2024-cyrus2024": "1GOnGrCj5Af8EV46JOzp5PVKnDMn-p5Vk",
    "robocin2024-aeteam2024": "1W1SLn5jrZqCfY5iXqXhIArPVvujbRItW",
    "r2d2-yushan2024": "1jsd1P014V-dF_R5hbh_Sv1x_HpzK7JiP",
    "r2d2-robocin2024": "1HoXcJFZtBoJdiYZVw0PNLIJiikWhXYBn",
    "r2d2-oxsy2024": "1hK0qrV5bFXGj689DYD67mHXXdfAPjfU9",
    "r2d2-mars2024": "1UoDTa2QjWucGQLGEG57uS_Y4mCR4W55Q",
    "r2d2-itandroids2024": "1PlD5E6yrGpacypQj6XBb8kquAgvBYyiT",
    "r2d2-helios2024": "1aotjYXtZXw9H_2QL9RzSCExqOzQWxD1q",
    "r2d2-helios2023": "1PqIrYfJIj1Y2X7A-r4oqY-0Ex4Jexk69",
    "r2d2-helios2022": "18GoDPBqGs5copPOkusgA2iZmrl7BZ7To",
    "r2d2-fra2024": "1g5RwJ5wQIKpfY6Q0x78eliechTqg7Q92",
    "r2d2-cyrus2024": "18L_eRm1c_YMhcod3a9M3FFpSXRpPGlI0",
    "r2d2-aeteam2024": "1BFoxAzRi0hOIxQhn6HsmhoWG1Dz9VITD",
    "oxsy2024-yushan2024": "1AkssVtevFs8AFvO5e-AM2TuLAxbzuNZr",
    "oxsy2024-robocin2024": "1THGM4VMPx2JXt3g2aezOckb_0P4qgxYh",
    "oxsy2024-r2d2": "1y8z2s5D81RsqHn3J4KkDj0LcNYO2yP2I",
    "oxsy2024-mars2024": "1QmU8pW2NQUQR4CKIQct1WVbRbZUYPIwe",
    "oxsy2024-itandroids2024": "1dlclI6gJM61F7WTZO-U-Yx1w8obgFTTN",
    "oxsy2024-helios2024": "1f10J3pzAv7F6FL65BB6_c-d9LIZdxjFJ",
    "oxsy2024-helios2023": "1HWBvVPfqiPV1gmDgl2izHZubF0bA-x1M",
    "oxsy2024-helios2022": "1-nNMaaJvKX5ahnQXDgV91Vxowu57U6Et",
    "oxsy2024-fra2024": "15SaNty_xtvhwasTj5qy-_Sk8ccco8Bm3",
    "oxsy2024-cyrus2024": "1ZYXjGL9kxYuBw1ch2aDmW0EqxMiFdWZO",
    "oxsy2024-aeteam2024": "14mQGPPc_1Tb2y0YprsGlghy9j3Y6hssq",
    "mars2024-yushan2024": "1KdR8ol0Ly2KerdYOR3bQbJTNbIIBStRs",
    "mars2024-robocin2024": "1BqtzigkZuOMGIydz_47GBxYJeWzI5E9m",
    "mars2024-r2d2": "1Mcf9Yy5CQQSOUCSDBctdMdLRi49bOsuV",
    "mars2024-oxsy2024": "1GA9PV2VBgR4K6sRXtmbkV7WoKVdugtG4",
    "mars2024-itandroids2024": "1-vVeh2cLU8x1SjHDRJoLSGUcnDRD7tkU",
    "mars2024-helios2024": "1elrQ6wudfYYYWBCB7u1DomfOB3Sk3BaN",
    "mars2024-helios2023": "1Pu55bsgWK9Iwe_Jyt-ZYtQtj8PrOfm3v",
    "mars2024-helios2022": "1O_PcGrk7yobpsu8YQ4j3iV7WF8jutDq2",
    "mars2024-fra2024": "1zyvDubd8K8tn32r7DPSEhVRf0VtSzpD4",
    "mars2024-cyrus2024": "152L0xHfnphkjfSeqpeLQ9cHo0Tw621G_",
    "mars2024-aeteam2024": "1vZtDnStDvrZn1LOePN-sx4dZRN3lkt-j",
    "helios2024-yushan2024": "1a1qvhDP1S4byzpXvSIK78SRmBIHyw5dT",
    "helios2024-robocin2024": "175QD4uDSs-E7hvuRa7C9QDzwr8eTxvmV",
    "helios2024-r2d2": "1KWM8AG7JxfOiQfvfFp03E2fHetWTxC-4",
    "helios2024-oxsy2024": "1Cvw17vcBG8zd_VDVRqO-d_Y6OGRHMomH",
    "helios2024-mars2024": "1Hqm9cDGztdeBR0Qka_bN3CZfFFQbs8Dz",
    "helios2024-itandroids2024": "1n-UrWI7oWtvL5pFRXJ2ZOIeimm1o7niK",
    "helios2024-helios2023": "1cjGIvrT7kA2fjZlxrS85NPDTKCOMqUvU",
    "helios2024-helios2022": "1nZsRcefTnW6w9iA8UrnhqJxyd39uyCIs",
    "helios2024-fra2024": "1JgiCROB6qMiNAWs-_7EtCYKHVOxNSnx-",
    "helios2024-cyrus2024": "15jrRCRS4itaI8J6lsMnn4-s00TZMW9b6",
    "helios2024-aeteam2024": "1WSdVAbXCWLwCh88AQALwjayPB_LthCYD",
    "helios2023-yushan2024": "1TzfrOmZIYU_KEgsoZUShzEPWQYXPbKWn",
    "helios2023-robocin2024": "15nELtXnJpETXxJbkcZsG_EFdl_loHtYr",
    "helios2023-r2d2": "1gRvrSwzAb-UdNGFD45Yj15o1iHnAU_0x",
    "helios2023-oxsy2024": "15PlaTP959Ndu8urEquUIBqx6d9B1mCBI",
    "helios2023-mars2024": "1rsNfusy4RSRLKWbeIIOIaDsHlL4aKZZx",
    "helios2023-itandroids2024": "1nJs3O9RgZV_o6dqGuT0QUYLxBl8VUuRB",
    "helios2023-helios2024": "1KvKwAvEPsOtq8PvRT8p9qthjg9yt3afo",
    "helios2023-helios2022": "1ZBE1YDFAsPMsPHzRYpObA7vxUdBT0-_S",
    "helios2023-fra2024": "1sae8dBWkvcq2Jn1UK2kA8chUfvsTFm3M",
    "helios2023-cyrus2024": "1szWS8cnEvrxH4hgWTL51elclGh4Pc2uc",
    "helios2023-aeteam2024": "1CUvhdsxQoV45-Ym6f0HsfAVvb9tyf_qx",
    "helios2022-yushan2024": "1Qsn1ecRWCvMjfXD3N_-pbJF7G_M6SnbT",
    "helios2022-robocin2024": "1akmWU7mEjfWXMhDiNfhI3AZqQgxBPi_Z",
    "helios2022-r2d2": "1NG8X6EdeSqKkVO7mDiZQ460mgxI8x649",
    "helios2022-oxsy2024": "1El1xKMPsfnPWIAzeTBL_u7WsJAB23MHb",
    "helios2022-mars2024": "1VVtIz3QfKl1I4hocbZuyuwQnO2KE7Dtw",
    "helios2022-itandroids2024": "1_GaHbxDbHJrSaFVOY7NHnjnmAmgA6uxI",
    "helios2022-helios2024": "1KZ0AFxzS3hkhm4EP-uBjlk2q_N6Lu3Gx",
    "helios2022-helios2023": "1Gpe-Q-TUStsvPPrvg78NucqF8jwL_YfC",
    "helios2022-fra2024": "1gshMh4AIjWb_vgeFm5HM4lx5jVeYICSU",
    "helios2022-cyrus2024": "1Gz9lFo6ihSjEKe2Bl36QotSn8wNJ6kVL",
    "helios2022-aeteam2024": "1V3k2FK71ZHhyhb7AWun2qAFZs1wl3iWj",
    "fra2024-yushan2024": "1g24Wd_BB5N-tXh75Ij0Yvals9_zg2LvJ",
    "fra2024-robocin2024": "1lFlZ3OWizP7A-d-QZYhCDTXgQKT7qzfb",
    "fra2024-r2d2": "1AfWSVmrD1wAif5lLgoXupuGulh_GIRAM",
    "fra2024-oxsy2024": "13vEWhQ8qHQsw_Xb1ARXf7NbzHHLuJ_YH",
    "fra2024-mars2024": "1GRz7QqDG1dN1T54L4dQmkLl1bgv-0iXu",
    "fra2024-itandroids2024": "1_BNSeWxxbyqSJzSSt4VB4RcDI23YNzLP",
    "fra2024-helios2024": "1vDfZ5FKI3JQv65smkltqH-z1_qHQj92c",
    "fra2024-helios2023": "1Frs2SV7Du5pScFSoFMfM8IB0H42YodoZ",
    "fra2024-helios2022": "161EmZQWWJygT7K-bM3gLkhBS7k0jp8FI",
    "fra2024-cyrus2024": "1gk88kmCJrlodggGd2vnmJRlIlIitMBzT",
    "fra2024-aeteam2024": "1CkabRVOpBJA6Cy0nX0Q8VfdxMhXhPP3q",
    "itandroids2024-yushan2024": "11kCE4Ea70kc81vVLKh0C-Lzzjd4p2hHi",
    "itandroids2024-robocin2024": "10BltsYE6X6FmChVBW8_woT5WGjBGtE_6",
    "itandroids2024-r2d2": "1OWhcn-qW8nKSFnoO-pG83S4U7x9kboCc",
    "itandroids2024-oxsy2024": "19kOPI4DNFTXSHHCQZHV-PKj1WSvDhUM4",
    "itandroids2024-mars2024": "1yYl1HitIN4dcdZbwl1ioQ60_8cxeRdPR",
    "itandroids2024-helios2024": "1wL-jFV7AQbcOvT6w_TLHoQ7cgM3C-lqW",
    "itandroids2024-helios2023": "1hDwQNo75vWZSDHcuWaT5LTBHR4wiOUxm",
    "itandroids2024-helios2022": "1Sp-rO0K3_FsRvYGTGU2TEKsxytvYeXV7",
    "itandroids2024-fra2024": "10spR8v9c1IZXchyvXRw79cGqc4PnWrEP",
    "itandroids2024-cyrus2024": "1VX8ibYmHYel0TGlFbKBBRFXi0Bv4xWlx",
    "itandroids2024-aeteam2024": "13zLiSOl-dejUybeAslEGnYS8SOaz120R",
    "cyrus2024-yushan2024": "1zxNILzpt2h8aYvPxX9ivwumSKa7WEbMh",
    "cyrus2024-robocin2024": "1zzwbgQXhu4-26IY0nIRW-f_Xrqjgig6B",
    "cyrus2024-r2d2": "18PZyPm0Z9j528mvX2tYjjWzo5iynn3QZ",
    "cyrus2024-oxsy2024": "11vKrYGzMDMb5A0KcHoDcARTJsLCMLK8Q",
    "cyrus2024-mars2024": "1P7TUigz0PaxlK3r4I1mVpAZY0vusDc7G",
    "cyrus2024-itandroids2024": "1NH-_d_OwZLN8ookVOKbjXBt6fAp00YEc",
    "cyrus2024-helios2024": "1T3TDS_UKnC1R8DFHrqOm1UXnz0klPpVX",
    "cyrus2024-helios2023": "1ZoeK2tEY-i-93aVGelN8s-9xvDiU8Faa",
    "cyrus2024-helios2022": "1IvqzJRvz7ceQAnM-XYnYWY1lS4wD9jWv",
    "cyrus2024-fra2024": "1-cMRGr0YzvsCjIxBJLfAgdz5xDUWp0ha",
    "cyrus2024-aeteam2024": "1ESwk_1wnOf7k-Tv7BkAI5jghg_ottANt",
    "aeteam2024-yushan2024": "1T6Vp5H4ESPJs3woHCPmgddYfkve_RDtP",
    "aeteam2024-robocin2024": "1KMZluJZZWUAAIKibw37OpzrDMH5NpqGt",
    "aeteam2024-r2d2": "1zyhGI2AukXoIZgrxz55b2I9XtNErdcri",
    "aeteam2024-oxsy2024": "1SWcmElKcCWIPrOVFurV2ZtnVUSz6Ga_q",
    "aeteam2024-mars2024": "165lU-4V7kWb-QA7GDW4aHS7LdFraH1WV",
    "aeteam2024-itandroids2024": "1_mTqEGVfXaNdkS8B0p0vuFEETkUC0_xl",
    "aeteam2024-helios2024": "1xkAaCJww11uBKz9EHM-h0ykf9zkT6Ae7",
    "aeteam2024-helios2023": "1ockvFRxjNGc38pWXqVP1SdCJOrZL6Mwa",
    "aeteam2024-helios2022": "11KBcpMClSljMyfnktipMrrU8WWYVfKjE",
    "aeteam2024-fra2024": "1cXRkbTZ8C6fLnLw6wpLAxRGRm2ZM-Y47",
    "aeteam2024-cyrus2024": "1Y-PAlWHHw-Xw-3MKIS9fp2VD-N6gMLNl",
}

def download_with_gdown(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination, quiet=False)

def extract_tracking_csv(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extracted = False
        for file in zip_ref.namelist():
            if file.lower().endswith('tracking.csv'):
                zip_ref.extract(file, extract_dir)
                extracted = True
                print(f"Extracted {file} to {extract_dir}")
        if not extracted:
            print(f"No tracking.csv found in {zip_path}")

for subpath in args.subpaths:
    file_id = subpath_to_file_id.get(subpath)
    if file_id is None:
        print(f"Error: No file ID found for subpath '{subpath}'")
        continue

    zip_name = f"{subpath}.zip"
    zip_path = os.path.join(args.save_dir, zip_name)

    try:
        download_with_gdown(file_id, zip_path)
        print(f"Downloaded {zip_name} successfully.")
        extract_tracking_csv(zip_path, args.save_dir)
    except Exception as e:
        print(f"Failed to download or process {zip_name}: {e}")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Deleted {zip_name}")