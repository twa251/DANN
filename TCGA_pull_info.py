# TCGA WSI path = "/scratch/wang_lab/BRCA_project/Data/TCGA/TCGA_BRCA"
# Check Size and scale for svs image file

import os
import pandas as pd
from openslide import OpenSlide

TCGA_path = "/scratch/wang_lab/BRCA_project/Data/TCGA/TCGA_BRCA"
output_path = os.path.join("/scratch/wang_lab/BRCA_project/Data/TCGA", "TCGA_size_scale.csv")

file_info_list= []
for dix, folder_name in enumerate(os.listdir(base_path)):
  folder_path = os.path.join(TCGA_path, folder_name)

if os.path.isdir(folder_path):
  for file_name in os.listdir(folder_path):
    if file_name.endwith('.svs'):
      file_path = os.path.join(folder_path,file_name)
      try:
        slide = OpenSlide(file_path)
        width, height = slide.dimensions
        
        #extract microns per pixel ( scale)
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0))

        file_info_list.append({
          'ID':idx,
          'File_Name': file_name,
          'Scale (um/px)': f"{mpp_x:.4f} x {mpp_y:.4f}",
          'Size (px)': f"{width} x {height}"
        })
      except Exception as e:

        file_info_list.append({
          'ID': idx,
          'File_Name': file_name,
          'Scale (um/px)': 'Error',
          'Size (px)': 'Error',
          'Error Message': str(e)
        })

file_info = pd.DataFrame(file_info_list)

file_info_df.to_csv(output_path, index = False)


  
