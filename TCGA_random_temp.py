import os
import random
import shutil

def assign_wsi_to_classes(input_dir, output_dir, num_classes=10, mapping_file="wsi_class_mapping.txt"):
    """
    Assigns each WSI folder to a random dummy class, writes the mapping to a text file,
    and copies the files into folders based on the assigned dummy class.

    Parameters:
        input_dir (str): Path to the directory containing WSI folders.
        output_dir (str): Path to the directory where files will be copied, organized by dummy classes.
        num_classes (int): Number of dummy classes to assign.
        mapping_file (str): Name of the output text file to save the mapping.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of WSI folders
    wsi_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Randomly assign each WSI folder to one of the dummy classes
    wsi_to_class = {wsi: random.randint(0, num_classes - 1) for wsi in wsi_folders}
    
    # Path for the mapping file
    mapping_file_path = os.path.join(output_dir, mapping_file)
    
    # Write the mapping to a text file
    with open(mapping_file_path, 'w') as f:
        for wsi, dummy_class in wsi_to_class.items():
            f.write(f"{wsi}: {dummy_class}\n")
    
    print(f"WSI-to-class mapping saved to {mapping_file_path}")
    
    # Copy files into class-based folders
    for wsi, dummy_class in wsi_to_class.items():
        source_folder = os.path.join(input_dir, wsi)
        class_folder = os.path.join(output_dir, f"class_{dummy_class}")
        
        # Ensure class folder exists
        os.makedirs(class_folder, exist_ok=True)
        
        # Copy all files from the WSI folder to the class folder
        for file_name in os.listdir(source_folder):
            source_file = os.path.join(source_folder, file_name)
            if os.path.isfile(source_file):  # Ensure it's a file
                shutil.copy(source_file, class_folder)
    
    print(f"Files have been copied to class folders in {output_dir}")

if __name__ == "__main__":
    input_directory = "/scratch/wang_lab/BRCA_project/DANN_BRCA/data/DANN_ALL/TCGA/TCGA_BRCA_temp_tw"  # Replace with your input directory
    output_directory = "/scratch/wang_lab/BRCA_project/DANN_BRCA/data/DANN_ALL/TCGA/TCGA_BRCA_dummy_class"  # Replace with your output directory
    assign_wsi_to_classes(input_directory, output_directory, num_classes=10, output_file="wsi_class_mapping.txt")
