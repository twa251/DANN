import os
import random

def assign_wsi_to_classes(input_dir, output_dir, num_classes=10, output_file="wsi_class_mapping.txt"):
    """
    Assigns each WSI folder to a random dummy class and writes the mapping to a text file.

    Parameters:
        input_dir (str): Path to the directory containing WSI folders.
        output_dir (str): Path to the directory where the output file will be saved.
        num_classes (int): Number of dummy classes to assign.
        output_file (str): Name of the output text file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of WSI folders
    wsi_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Randomly assign each WSI folder to one of the dummy classes
    wsi_to_class = {wsi: random.randint(0, num_classes - 1) for wsi in wsi_folders}
    
    # Path for the output file
    output_path = os.path.join(output_dir, output_file)
    
    # Write the mapping to a text file
    with open(output_path, 'w') as f:
        for wsi, dummy_class in wsi_to_class.items():
            f.write(f"{wsi}: {dummy_class}\n")
    
    print(f"WSI-to-class mapping saved to {output_path}")

    return wsi_to_class

# Example usage
if __name__ == "__main__":
    input_directory = "/scratch/wang_lab/BRCA_project/DANN_BRCA/data/DANN_ALL/TCGA/TCGA_BRCA_temp_tw"  # Replace with your input directory
    output_directory = "/scratch/wang_lab/BRCA_project/DANN_BRCA/data/DANN_ALL/TCGA/TCGA_BRCA_dummy_class"  # Replace with your output directory
    assign_wsi_to_classes(input_directory, output_directory, num_classes=10, output_file="wsi_class_mapping.txt")
