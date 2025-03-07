import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, exposure, util
from scipy import ndimage
import pyvista as pv

def process_mri_slices(image_paths, output_folder):
    """
    Process MRI slices to create a 3D model from just 3 planes with improved tissue differentiation.
    
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
        
        # Normalize image to 0-1 range
        img_norm = img.copy()
        if img_norm.max() > 0:
            img_norm = img_norm / img_norm.max()
        
        # Enhance contrast to better distinguish tissues
        img_eq = exposure.equalize_adapthist(img_norm, clip_limit=0.03)
        
        # Get histogram for analysis
        hist, hist_centers = exposure.histogram(img_eq)
        
        # Define thresholds based on intensity for different tissues
        # Pure black (background/outline) - lowest intensity
        black_threshold = 0.1  # Adjust based on your image
        
        # Grey (bone) - mid-range intensity
        bone_threshold_low = 0.3  # Adjust based on your image
        bone_threshold_high = 0.65
        
        # Slightly darker white (soft tissue) - higher range
        soft_threshold_low = 0.65
        soft_threshold_high = 0.85
        
        # White (muscle) - highest intensity
        muscle_threshold_low = 0.85
        
        # Create masks for each tissue type
        black_mask = img_eq <= black_threshold
        bone_mask = (img_eq > bone_threshold_low) & (img_eq < bone_threshold_high)
        soft_tissue_mask = (img_eq >= soft_threshold_low) & (img_eq < soft_threshold_high)
        muscle_mask = img_eq >= muscle_threshold_low
        
        # Clean up masks with morphological operations
        # Remove small objects and fill holes
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=50)
        bone_mask = morphology.closing(bone_mask, morphology.disk(2))
        
        soft_tissue_mask = morphology.remove_small_objects(soft_tissue_mask, min_size=100)
        soft_tissue_mask = morphology.closing(soft_tissue_mask, morphology.disk(3))
        
        muscle_mask = morphology.remove_small_objects(muscle_mask, min_size=50)
        muscle_mask = morphology.closing(muscle_mask, morphology.disk(2))
        
        # Ensure no overlap between masks (prioritizing in order: muscle, soft tissue, bone)
        muscle_mask_final = muscle_mask
        soft_tissue_mask_final = soft_tissue_mask & ~muscle_mask_final
        bone_mask_final = bone_mask & ~soft_tissue_mask_final & ~muscle_mask_final
        
        # Save segmentation results
        segmentations[plane] = {
            'bone': bone_mask_final,
            'soft_tissue': soft_tissue_mask_final,
            'muscle': muscle_mask_final
        }
        
        # Create visualization of segmentation
        # Combined visualization - Blue for bone, Green for soft tissue, Red for muscle
        combined = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        combined[bone_mask_final, 2] = 255  # Blue for bone
        combined[soft_tissue_mask_final, 1] = 255  # Green for soft tissue
        combined[muscle_mask_final, 0] = 255  # Red for muscle
        
        # Save combined visualization
        combined_output = os.path.join(output_folder, 'segmented', f"{plane}_segmentation.png")
        io.imsave(combined_output, combined)
        
        # Show original and segmentation results
        plt.figure(figsize=(20, 5))
        plt.subplot(151)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original {plane}')
        
        plt.subplot(152)
        plt.imshow(img_eq, cmap='gray')
        plt.title('Enhanced Image')
        
        plt.subplot(153)
        plt.imshow(bone_mask_final, cmap='Blues')
        plt.title('Bone (Grey Regions)')
        
        plt.subplot(154)
        plt.imshow(soft_tissue_mask_final, cmap='Greens')
        plt.title('Soft Tissue (Darker White)')
        
        plt.subplot(155)
        plt.imshow(muscle_mask_final, cmap='Reds')
        plt.title('Muscle (White Regions)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'segmented', f"{plane}_result.png"))
        
        # Additional visualization showing combined segmentation
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original {plane}')
        
        plt.subplot(122)
        plt.imshow(combined)
        plt.title('Combined Segmentation\nBlue: Bone, Green: Soft Tissue, Red: Muscle')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'segmented', f"{plane}_combined.png"))
        plt.close('all')
    
    # Create pseudo-3D volume from the three planes
    create_3d_model_from_planes(segmentations, output_folder)
    
    return segmentations

def create_3d_model_from_planes(segmentations, output_folder):
    """
    Create a 3D model from three orthogonal planes by interpolation.
    Now includes separate modeling for bone, soft tissue, and muscle.
    
    Args:
        segmentations: Dictionary with segmentation results for each plane
        output_folder: Where to save the 3D models
    """
    # Check if we have all three planes
    required_planes = ['axial', 'sagittal', 'coronal']
    if not all(plane in segmentations for plane in required_planes):
        print("Error: Need all three planes (axial, sagittal, coronal) for 3D reconstruction")
        return
    
    # Create volumes for each tissue type
    volume_size = 128
    bone_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    soft_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    muscle_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    
    # Place each plane in the volume - simplified approach
    # Axial plane (xy plane, varying z)
    axial_bone = segmentations['axial']['bone']
    axial_soft = segmentations['axial']['soft_tissue']
    axial_muscle = segmentations['axial']['muscle']
    
    # Resize to fit our volume
    axial_bone_resized = resize_binary_image(axial_bone, (volume_size, volume_size))
    axial_soft_resized = resize_binary_image(axial_soft, (volume_size, volume_size))
    axial_muscle_resized = resize_binary_image(axial_muscle, (volume_size, volume_size))
    
    # Place in middle of z-axis
    mid_z = volume_size // 2
    bone_volume[:, :, mid_z] = axial_bone_resized
    soft_volume[:, :, mid_z] = axial_soft_resized
    muscle_volume[:, :, mid_z] = axial_muscle_resized
    
    # Sagittal plane (yz plane, varying x)
    sagittal_bone = segmentations['sagittal']['bone']
    sagittal_soft = segmentations['sagittal']['soft_tissue']
    sagittal_muscle = segmentations['sagittal']['muscle']
    
    # Resize to fit our volume
    sagittal_bone_resized = resize_binary_image(sagittal_bone, (volume_size, volume_size))
    sagittal_soft_resized = resize_binary_image(sagittal_soft, (volume_size, volume_size))
    sagittal_muscle_resized = resize_binary_image(sagittal_muscle, (volume_size, volume_size))
    
    # Place in middle of x-axis
    mid_x = volume_size // 2
    bone_volume[mid_x, :, :] = sagittal_bone_resized.T
    soft_volume[mid_x, :, :] = sagittal_soft_resized.T
    muscle_volume[mid_x, :, :] = sagittal_muscle_resized.T
    
    # Coronal plane (xz plane, varying y)
    coronal_bone = segmentations['coronal']['bone']
    coronal_soft = segmentations['coronal']['soft_tissue']
    coronal_muscle = segmentations['coronal']['muscle']
    
    # Resize to fit our volume
    coronal_bone_resized = resize_binary_image(coronal_bone, (volume_size, volume_size))
    coronal_soft_resized = resize_binary_image(coronal_soft, (volume_size, volume_size))
    coronal_muscle_resized = resize_binary_image(coronal_muscle, (volume_size, volume_size))
    
    # Place in middle of y-axis
    mid_y = volume_size // 2
    bone_volume[:, mid_y, :] = coronal_bone_resized
    soft_volume[:, mid_y, :] = coronal_soft_resized
    muscle_volume[:, mid_y, :] = coronal_muscle_resized
    
    # Interpolate between planes (simple dilation approach)
    bone_volume = ndimage.binary_dilation(bone_volume, iterations=3)
    soft_volume = ndimage.binary_dilation(soft_volume, iterations=3)
    muscle_volume = ndimage.binary_dilation(muscle_volume, iterations=3)
    
    # Smooth the volumes
    bone_volume_smooth = ndimage.gaussian_filter(bone_volume.astype(float), sigma=1.0)
    soft_volume_smooth = ndimage.gaussian_filter(soft_volume.astype(float), sigma=1.0)
    muscle_volume_smooth = ndimage.gaussian_filter(muscle_volume.astype(float), sigma=1.0)
    
    # Convert back to binary with appropriate thresholds
    bone_volume = bone_volume_smooth > 0.2
    soft_volume = soft_volume_smooth > 0.2
    muscle_volume = muscle_volume_smooth > 0.2
    
    # Ensure no overlap between volumes (prioritize muscle, then soft tissue, then bone)
    muscle_volume_final = muscle_volume
    soft_volume_final = soft_volume & ~muscle_volume_final
    bone_volume_final = bone_volume & ~soft_volume_final & ~muscle_volume_final
    
    # Create meshes using marching cubes algorithm
    print("Creating 3D meshes using marching cubes...")
    
    # Bone mesh
    try:
        bone_verts, bone_faces, _, _ = measure.marching_cubes(bone_volume_final)
        bone_mesh = pv.PolyData(bone_verts, faces=np.column_stack(([3] * len(bone_faces), bone_faces)))
        bone_mesh = bone_mesh.smooth(n_iter=15)
        bone_mesh.save(os.path.join(output_folder, 'models', 'bone_model.stl'))
        print("Saved bone model")
    except Exception as e:
        print(f"Error creating bone mesh: {e}")
    
    # Soft tissue mesh
    try:
        soft_verts, soft_faces, _, _ = measure.marching_cubes(soft_volume_final)
        soft_mesh = pv.PolyData(soft_verts, faces=np.column_stack(([3] * len(soft_faces), soft_faces)))
        soft_mesh = soft_mesh.smooth(n_iter=15)
        soft_mesh.save(os.path.join(output_folder, 'models', 'soft_tissue_model.stl'))
        print("Saved soft tissue model")
    except Exception as e:
        print(f"Error creating soft tissue mesh: {e}")
    
    # Muscle mesh
    try:
        muscle_verts, muscle_faces, _, _ = measure.marching_cubes(muscle_volume_final)
        muscle_mesh = pv.PolyData(muscle_verts, faces=np.column_stack(([3] * len(muscle_faces), muscle_faces)))
        muscle_mesh = muscle_mesh.smooth(n_iter=15)
        muscle_mesh.save(os.path.join(output_folder, 'models', 'muscle_model.stl'))
        print("Saved muscle model")
    except Exception as e:
        print(f"Error creating muscle mesh: {e}")
    
    # Create visualization of the 3D models
    try:
        p = pv.Plotter(off_screen=True)
        p.add_mesh(bone_mesh, color='blue', opacity=0.7, label='Bone (Grey)')
        p.add_mesh(soft_mesh, color='green', opacity=0.5, label='Soft Tissue (Darker White)')
        p.add_mesh(muscle_mesh, color='red', opacity=0.6, label='Muscle (White)')
        p.add_legend()
        p.camera_position = 'iso'
        p.screenshot(os.path.join(output_folder, 'models', '3d_visualization.png'))
        
        # Create a separate visualization for each tissue type
        for tissue, mesh, color in [
            ('bone', bone_mesh, 'blue'),
            ('soft_tissue', soft_mesh, 'green'),
            ('muscle', muscle_mesh, 'red')
        ]:
            p_single = pv.Plotter(off_screen=True)
            p_single.add_mesh(mesh, color=color, opacity=1.0)
            p_single.camera_position = 'iso'
            p_single.screenshot(os.path.join(output_folder, 'models', f'{tissue}_visualization.png'))
            
        print("Created 3D visualizations")
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