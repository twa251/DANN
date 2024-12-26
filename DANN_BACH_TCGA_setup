# improt package

import os
import numpy as np
import random
from PIL import Image
import argparse

def process_npy_to_img(source_root, target_root, stages):
    for stage in stages:
        source_folder=os.path.join(source_root, stage)
        target_folder=os.path.join(target_root, stage)

        #create target folder if not exists
        os.makedirs(target_folder, exist_ok=True)

        for file_name in os.listdir(source_folder):
            if file_name.endswith('.npy'):
                source_file=os.path.join(source_folder, file_name)
                target_file=os.path.join(target_folder, file_name.replace("_norm.npy",".tif"))

                array = np.load(source_file)
                tile=Image.fromarray(array)
                tile.save(target_file,format="TIFF")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", help="Path to folder containing subfolders with .npy files",
                        default='/scratch/wang_lab/BRCA_project/DANN_BRCA/data/DANN_ALL/BACH/ICIAR2018_BACH_Challenge/Photos_crop')
    parser.add_argument("--target_root", help="Path to folder where .tif files will be saved",
                        default='/scratch/wang_lab/BRCA_project/DANN_BRCA/data/DANN_BRCA_img')
    parser.add_argument("--stages", nargs='+', help="List of stages to process",
                        default=['Normal', 'Invasive', 'InSitu', 'Benign'])
    args = parser.parse_args()

    # Pass the parsed arguments to the processing function
    process_npy_to_img(args.source_root, args.target_root, args.stages)


if __name__ == '__main__':
    main()
