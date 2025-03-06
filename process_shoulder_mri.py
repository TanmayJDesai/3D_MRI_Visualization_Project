import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, exposure
from scipy import ndimage
import pyvista as pv

def process_mri_slices(image_paths, output_folder):
    """
    Process MRI slices to create a 3D model from just 3 planes.
    
    Args:
        image_paths: Dictionary with keys 'axial', 'sagittal', 'coronal' and paths as values
        output_folder: Where to save the results
    """
    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'segmented'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'models'), exist_ok=True)
    
    # Dictionary to store segmentation results
    segmentations = {}
    
    # Process each plane
    for plane, img_path in image_paths.items():
        print(f"Processing {plane} plane image: {img_path}")
        
        # Load image
        if not os.path.exists(img_path):
            print(f"Error: Image file not found: {img_path}")
            continue
            
        img = io.imread(img_path, as_gray=True)
        
        # Enhance contrast
        img_eq = exposure.equalize_hist(img)
        
        # Segment bone (bright in MRI)
        bone_threshold = filters.threshold_otsu(img_eq)
        bone_mask = img_eq > bone_threshold * 1.2  # Adjust multiplier based on your images
        
        # Clean up the bone mask
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=50)
        bone_mask = morphology.closing(bone_mask, morphology.disk(2))
        
        # Segment soft tissue (muscle, tendons - medium intensity in MRI)
        soft_threshold_low = bone_threshold * 0.5  # Adjust based on your images
        soft_threshold_high = bone_threshold * 0.9
        soft_tissue_mask = (img_eq > soft_threshold_low) & (img_eq < soft_threshold_high)
        
        # Clean up the soft tissue mask
        soft_tissue_mask = morphology.remove_small_objects(soft_tissue_mask, min_size=100)
        soft_tissue_mask = morphology.closing(soft_tissue_mask, morphology.disk(3))
        
        # Save segmentation results
        segmentations[plane] = {
            'bone': bone_mask,
            'soft_tissue': soft_tissue_mask
        }
        
        # Save visualization of segmentation
        # Combined visualization
        combined = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        combined[bone_mask, 0] = 255  # Red for bone
        combined[soft_tissue_mask, 1] = 255  # Green for soft tissue
        
        combined_output = os.path.join(output_folder, 'segmented', f"{plane}_segmentation.png")
        io.imsave(combined_output, combined)
        
        # Show original and segmentation
        plt.figure(figsize=(15, 5))
        plt.subplot(141)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original {plane}')
        
        plt.subplot(142)
        plt.imshow(bone_mask, cmap='gray')
        plt.title('Bone Segmentation')
        
        plt.subplot(143)
        plt.imshow(soft_tissue_mask, cmap='gray')
        plt.title('Soft Tissue Segmentation')
        
        plt.subplot(144)
        plt.imshow(combined)
        plt.title('Combined Segmentation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'segmented', f"{plane}_result.png"))
        plt.close()
    
    # Create pseudo-3D volume from the three planes
    # Since we only have 3 planes, we'll create a simplified 3D model by stacking and interpolating
    create_3d_model_from_planes(segmentations, output_folder)
    
    return segmentations

def create_3d_model_from_planes(segmentations, output_folder):
    """
    Create a 3D model from three orthogonal planes by interpolation.
    
    This is a simplified approach since we only have 3 images.
    
    Args:
        segmentations: Dictionary with segmentation results for each plane
        output_folder: Where to save the 3D models
    """
    # Check if we have all three planes
    required_planes = ['axial', 'sagittal', 'coronal']
    if not all(plane in segmentations for plane in required_planes):
        print("Error: Need all three planes (axial, sagittal, coronal) for 3D reconstruction")
        return
    
    # Create a simple volume
    # We'll use a small volume size for demonstration, adjust as needed
    volume_size = 128
    bone_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    soft_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    
    # Place each plane in the volume
    # This is a simplified approach - in a real project, you'd need proper registration
    
    # Axial plane (xy plane, varying z)
    axial_bone = segmentations['axial']['bone']
    axial_soft = segmentations['axial']['soft_tissue']
    
    # Resize to fit our volume
    axial_bone_resized = resize_binary_image(axial_bone, (volume_size, volume_size))
    axial_soft_resized = resize_binary_image(axial_soft, (volume_size, volume_size))
    
    # Place in middle of z-axis
    mid_z = volume_size // 2
    bone_volume[:, :, mid_z] = axial_bone_resized
    soft_volume[:, :, mid_z] = axial_soft_resized
    
    # Sagittal plane (yz plane, varying x)
    sagittal_bone = segmentations['sagittal']['bone']
    sagittal_soft = segmentations['sagittal']['soft_tissue']
    
    # Resize to fit our volume
    sagittal_bone_resized = resize_binary_image(sagittal_bone, (volume_size, volume_size))
    sagittal_soft_resized = resize_binary_image(sagittal_soft, (volume_size, volume_size))
    
    # Place in middle of x-axis
    mid_x = volume_size // 2
    bone_volume[mid_x, :, :] = sagittal_bone_resized.T
    soft_volume[mid_x, :, :] = sagittal_soft_resized.T
    
    # Coronal plane (xz plane, varying y)
    coronal_bone = segmentations['coronal']['bone']
    coronal_soft = segmentations['coronal']['soft_tissue']
    
    # Resize to fit our volume
    coronal_bone_resized = resize_binary_image(coronal_bone, (volume_size, volume_size))
    coronal_soft_resized = resize_binary_image(coronal_soft, (volume_size, volume_size))
    
    # Place in middle of y-axis
    mid_y = volume_size // 2
    bone_volume[:, mid_y, :] = coronal_bone_resized
    soft_volume[:, mid_y, :] = coronal_soft_resized
    
    # Interpolate between planes (simple dilation approach)
    bone_volume = ndimage.binary_dilation(bone_volume, iterations=3)
    soft_volume = ndimage.binary_dilation(soft_volume, iterations=3)
    
    # Smooth the volumes
    bone_volume_smooth = ndimage.gaussian_filter(bone_volume.astype(float), sigma=1.0)
    soft_volume_smooth = ndimage.gaussian_filter(soft_volume.astype(float), sigma=1.0)
    
    # Convert back to binary
    bone_volume = bone_volume_smooth > 0.2
    soft_volume = soft_volume_smooth > 0.2
    
    # Create meshes using marching cubes algorithm
    print("Creating 3D meshes using marching cubes...")
    
    # Bone mesh
    try:
        bone_verts, bone_faces, _, _ = measure.marching_cubes(bone_volume)
        bone_mesh = pv.PolyData(bone_verts, faces=np.column_stack(([3] * len(bone_faces), bone_faces)))
        bone_mesh = bone_mesh.smooth(n_iter=15)
        bone_mesh.save(os.path.join(output_folder, 'models', 'bone_model.stl'))
        print("Saved bone model")
    except Exception as e:
        print(f"Error creating bone mesh: {e}")
    
    # Soft tissue mesh
    try:
        soft_verts, soft_faces, _, _ = measure.marching_cubes(soft_volume)
        soft_mesh = pv.PolyData(soft_verts, faces=np.column_stack(([3] * len(soft_faces), soft_faces)))
        soft_mesh = soft_mesh.smooth(n_iter=15)
        soft_mesh.save(os.path.join(output_folder, 'models', 'soft_tissue_model.stl'))
        print("Saved soft tissue model")
    except Exception as e:
        print(f"Error creating soft tissue mesh: {e}")
    
    # Create visualization of the 3D models
    try:
        p = pv.Plotter(off_screen=True)
        p.add_mesh(bone_mesh, color='white', opacity=1.0)
        p.add_mesh(soft_mesh, color='red', opacity=0.5)
        p.camera_position = 'iso'
        p.screenshot(os.path.join(output_folder, 'models', '3d_visualization.png'))
        print("Created 3D visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def resize_binary_image(binary_image, target_size):
    """Resize a binary image to a target size"""
    from skimage.transform import resize
    resized = resize(binary_image.astype(float), target_size, order=0, mode='constant', anti_aliasing=False)
    return resized > 0.5

if __name__ == "__main__":
    print("Script is starting...")
    
    # Define paths to your MRI images
    image_paths = {
        'axial': 'Images/axialA.JPG',
        'sagittal': 'Images/sagittalA.JPG',
        'coronal': 'Images/coronalA.JPG'
    }
    
    # Check if files exist
    for plane, path in image_paths.items():
        print(f"Checking if {path} exists: {os.path.exists(path)}")
    
    # Output folder
    output_folder = 'results'
    
    # Process the MRI slices
    print("Starting processing...")
    process_mri_slices(image_paths, output_folder)
    
    print("\nProcessing complete! Results saved to", output_folder)