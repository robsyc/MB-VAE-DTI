import glob
import json
import os
import shutil
import tarfile

import click
import requests
from tqdm import tqdm

from bmfm_sm.core.data_modules.splits import create_splits


def dir_is_empty(directory: str) -> bool:
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return True
    elif len(os.listdir(directory)) == 0:
        print(f"Directory {directory} is empty.")
        return True
    else:
        print(f"Directory {directory} is not empty.")
        return False


def create_split_json(output_dir, split_dir):
    print("Creating data splits JSON files.")
    frac_val = 0.1
    frac_test = 0.1

    os.makedirs(split_dir, exist_ok=True)

    if dir_is_empty(split_dir):
        file_all = sorted(glob.glob(os.path.join(output_dir, "*csv")))
        split_all = ["ligand_random_scaffold", "ligand_scaffold", "random"]

        for the_file in file_all:
            dataset = os.path.basename(the_file).split(".csv")[0]
            for the_split in split_all:
                if the_split == "ligand_scaffold_balanced":
                    frac_train = 0.8
                else:
                    frac_train = 1.0
                path_split = os.path.join(split_dir, the_split)
                os.makedirs(path_split, exist_ok=True)
                dict_split = create_splits(
                    source_file=the_file,
                    frac_train=frac_train,
                    frac_val=frac_val,
                    frac_test=frac_test,
                    strategy=the_split,
                )
                output_file = os.path.join(path_split, dataset + "_split.json")
                with open(output_file, "w") as json_file:
                    json.dump(dict_split, json_file)
        print("Split JSON files created successfully.")
    else:
        print(f"Skipping split creation as {split_dir} is not empty.")


def download_file_from_google_drive(file_id, destination, chunk_size=32768):
    url = "https://drive.usercontent.google.com/download?export=download"

    session = requests.Session()
    params = {"id": file_id, "confirm": "t"}
    response = session.get(url, params=params, stream=True)

    yield from save_response_content(response, destination, chunk_size)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination, chunk_size):
    with open(destination, "wb") as f:
        for i, chunk in enumerate(response.iter_content(chunk_size)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                yield i, chunk_size


def download_moleculenet_data(file_id, output_dir):
    tarball = os.path.join(output_dir, "MoleculeNet.tar.gz")
    if not os.path.exists(tarball):
        print("Downloading file...")
        with tqdm(total=56714, unit="B", unit_scale=True, desc="Downloading") as pbar:
            for i, chunk_size in download_file_from_google_drive(file_id, tarball):
                pbar.update(chunk_size)
        print("Download complete. Extracting tar.gz file.")
    else:
        print(f"Skipping download as {tarball} exists.")

    if not tarfile.is_tarfile(tarball):
        print(
            "Error: The downloaded file is not a valid tar.gz archive. Please check if the file ID is correct and that you have permission to access the file"
        )
        return None
    else:
        return tarball


def extract_csv_files_from_tar(tarball_path, output_dir, remove_tarball=False):
    print(f"Extracting files from {tarball_path} into {output_dir}")
    with tarfile.open(tarball_path, "r:gz") as tar:
        csv_members = [
            member for member in tar.getmembers() if member.name.endswith(".csv")
        ]
        tar.extractall(path=output_dir, members=csv_members)
        print(f"Extracted {len(csv_members)} CSV files to {output_dir}")
    if remove_tarball:
        os.remove(tarball_path)
        print(f"Removed file: {tarball_path}")


def clean_and_reorganize(molnet_raw_data_src_dir, dataset_path):
    print("Cleaning up the data.")
    processed_csv_files = glob.glob(
        os.path.join(molnet_raw_data_src_dir, "**", "processed", "*csv"), recursive=True
    )

    if processed_csv_files:
        for csv_file in processed_csv_files:
            dataset_name = os.path.basename(csv_file).split("_processed_ac")[0]
            new_csv_name = os.path.join(dataset_path, f"{dataset_name}.csv")
            shutil.move(csv_file, new_csv_name)
            print(f"Moved and renamed {csv_file} to {new_csv_name}")
        print(f"Removed directory: {molnet_raw_data_src_dir}")
        shutil.rmtree(molnet_raw_data_src_dir)
    else:
        print("No processed CSV files found.")


@click.command()
@click.option(
    "--file-id",
    default="1IdW6J6tX4j5JU0bFcQcuOBVwGNdX7pZp",
    help="Google Drive file ID.",
)
def main(file_id):
    if not os.environ.get("BMFM_HOME"):
        print(
            "Define BMFM_HOME environment variable, this will be the output dir for the data."
        )
        return

    output_dir = os.environ["BMFM_HOME"]
    datasets_dir = os.path.join(output_dir, "datasets")
    raw_data_dir = os.path.join(datasets_dir, "raw_data")
    splits_dir = os.path.join(datasets_dir, "splits")
    molnet_raw_data_src_dir = os.path.join(raw_data_dir, "MPP")
    molnet_raw_data_dir = os.path.join(raw_data_dir, "MoleculeNet")
    molnet_splits_dir = os.path.join(splits_dir, "MoleculeNet")
    os.makedirs(molnet_raw_data_dir, exist_ok=True)
    os.makedirs(molnet_splits_dir, exist_ok=True)

    if not os.listdir(molnet_raw_data_dir):
        tarball_path = download_moleculenet_data(file_id, raw_data_dir)
        extract_csv_files_from_tar(tarball_path, raw_data_dir)
        clean_and_reorganize(molnet_raw_data_src_dir, molnet_raw_data_dir)
    else:
        print(
            f"Data already exists in {molnet_raw_data_dir} —skipping download and cleanup."
        )

    create_split_json(molnet_raw_data_dir, molnet_splits_dir)


if __name__ == "__main__":
    main()
