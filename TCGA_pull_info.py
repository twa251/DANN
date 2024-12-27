# TCGA WSI path = "/scratch/wang_lab/BRCA_project/Data/TCGA/TCGA_BRCA"
# Check Size and scale for svs image file

import os
import pandas as pd
from openslide import OpenSlide
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image

# Base path for TCGA folders
TCGA_path = "/scratch/wang_lab/BRCA_project/Data/TCGA/TCGA_BRCA"
BACH_path = "/scratch/wang_lab/BRCA_project/Data/BACH/ICIAR2018_BACH_Challenge_TrainDataset/Photos"
#output_path_TCGA = os.path.join("/home/twa251", "TCGA_size_scale.csv")
#output_path_BACH = os.path.join("/home/twa251", "TCGA_size_scale.csv")
output_path_TCGA = os.path.join("/scratch/wang_lab/BRCA_project/Tracking_File", "TCGA_size_scale.csv")
output_path_BACH = os.path.join("/scratch/wang_lab/BRCA_project/Tracking_File", "BACH_size_scale.csv")

# Function to process a WSI file
def process_wsi_file(file_path):
    try:
        slide = OpenSlide(file_path)
        width, height = slide.dimensions
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0))
        return {
            'File Name': os.path.basename(file_path),
            'Scale (um/px)': f"{mpp_x:.4f} x {mpp_y:.4f}",
            'Size (px)': f"{width} x {height}"
        }
    except Exception as e:
        return {
            'File Name': os.path.basename(file_path),
            'Scale (um/px)': 'Error',
            'Size (px)': 'Error',
            'Error Message': str(e)
        }



# Collect all SVS file paths
def get_wsi_file_paths(base_path, extensions = ('.svs', '.tif', '.tiff')):
    wsi_files = []
    for folder_name in tqdm(os.listdir(base_path), desc="Collecting files"):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            wsi_files.extend([
                os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                if file_name.lower().endswith(extensions)
            ])
    return wsi_files



# Main function to process all files in parallel
def main():
    print("Processing TCGA files...")
    tcga_file_paths = get_wsi_file_paths(TCGA_path)

    print(f"Found {len(tcga_file_paths)} WSI files. Processing...")
    tcga_results = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Process files in parallel and show progress bar
        for result in tqdm(executor.map(process_wsi_file, tcga_file_paths), total=len(tcga_file_paths), desc="Processing TCGA WSI files"):
            tcga_results.append(result)
    # save TCGA results       
    TCGA_info_df = pd.DataFrame(tcga_results)
    TCGA_info_df.to_csv(output_path_TCGA, index=False)
    print(f"TCGA info saved {output_path_TCGA}")

    
    print(f"Processing BACH training files...")
    bach_file_paths = get_wsi_file_paths(BACH_path)
    print(f"Found {len(bach_file_paths)} WSI files. Processing...")
    bach_results = []

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_wsi_file, bach_file_paths), total=len(bach_file_paths), desc="Processing BACH WSI files"):
            bach_results.append(result)
            
    # save BACH results       
    BACH_info_df = pd.DataFrame(bach_results)
    BACH_info_df.to_csv(output_path_BACH, index=False)
    print(f"BACH info saved {output_path_BACH}")


if __name__ == "__main__":
    main()
