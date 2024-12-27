# TCGA WSI path = "/scratch/wang_lab/BRCA_project/Data/TCGA/TCGA_BRCA"
# Check Size and scale for svs image file

import os
import pandas as pd
from openslide import OpenSlide
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Base path for TCGA folders
TCGA_path = "/scratch/wang_lab/BRCA_project/Data/TCGA/TCGA_BRCA"
output_path = os.path.join("/home/twa251", "TCGA_size_scale.csv")

# Function to process a single SVS file
def process_svs_file(file_path):
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
def get_svs_file_paths(base_path):
    svs_files = []
    for folder_name in tqdm(os.listdir(base_path), desc="Collecting files"):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            svs_files.extend([
                os.path.join(folder_path, file_name)
                for file_name in os.listdir(folder_path)
                if file_name.endswith('.svs')
            ])
    return svs_files

# Main function to process all files in parallel
def main():
    print("Scanning folders and collecting SVS files...")
    svs_file_paths = get_svs_file_paths(TCGA_path)

    print(f"Found {len(svs_file_paths)} SVS files. Processing...")
    results = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Process files in parallel and show progress bar
        for result in tqdm(executor.map(process_svs_file, svs_file_paths), total=len(svs_file_paths), desc="Processing SVS files"):
            results.append(result)

    # Convert results to DataFrame
    file_info_df = pd.DataFrame(results)

    # Save results to CSV
    file_info_df.to_csv(output_path, index=False)
    print(f"Processing complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
