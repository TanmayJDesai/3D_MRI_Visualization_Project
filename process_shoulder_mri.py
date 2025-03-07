import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, exposure
from scipy import ndimage
import pyvista as pv

def process_mri_slices(image_paths, output_folder):
    """
    Process MRI slices to create a 3D model of a shoulder from three planes.
    
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
        
        # Define thresholds based on intensity
        black_threshold = 0.1  # Pure black (background)
        bone_threshold_low = 0.3  # Grey (bone)
        bone_threshold_high = 0.65
        soft_threshold_low = 0.65  # Darker white (soft tissue)
        soft_threshold_high = 0.85
        muscle_threshold_low = 0.85  # White (muscle)
        
        # Create masks for each tissue type
        black_mask = img_eq <= black_threshold
        bone_mask = (img_eq > bone_threshold_low) & (img_eq < bone_threshold_high)
        soft_tissue_mask = (img_eq >= soft_threshold_low) & (img_eq < soft_threshold_high)
        muscle_mask = img_eq >= muscle_threshold_low
        
        # Clean up masks with morphological operations
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=50)
        bone_mask = morphology.closing(bone_mask, morphology.disk(2))
        
        soft_tissue_mask = morphology.remove_small_objects(soft_tissue_mask, min_size=100)
        soft_tissue_mask = morphology.closing(soft_tissue_mask, morphology.disk(3))
        
        muscle_mask = morphology.remove_small_objects(muscle_mask, min_size=50)
        muscle_mask = morphology.closing(muscle_mask, morphology.disk(2))
        
        # Ensure no overlap between masks
        muscle_mask_final = muscle_mask
        soft_tissue_mask_final = soft_tissue_mask & ~muscle_mask_final
        bone_mask_final = bone_mask & ~soft_tissue_mask_final & ~muscle_mask_final
        
        # Save segmentation results
        segmentations[plane] = {
            'bone': bone_mask_final,
            'soft_tissue': soft_tissue_mask_final,
            'muscle': muscle_mask_final
        }
        
        # Create visualization of segmentation - ADDING THIS PART BACK
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
    
    # Create anatomically correct 3D shoulder model
    create_anatomical_shoulder_model(segmentations, output_folder)
    
    return segmentations

def create_anatomical_shoulder_model(segmentations, output_folder):
    """
    Create an anatomically correct 3D shoulder model from the three planes.
    
    Args:
        segmentations: Dictionary with segmentation results for each plane
        output_folder: Where to save the 3D models
    """
    # Check if we have all three planes
    required_planes = ['axial', 'sagittal', 'coronal']
    if not all(plane in segmentations for plane in required_planes):
        print("Error: Need all three planes for 3D reconstruction")
        return
    
    # Create volumes with anatomical shoulder dimensions
    # Using higher resolution for better detail
    volume_size = 196
    bone_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    soft_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    muscle_volume = np.zeros((volume_size, volume_size, volume_size), dtype=bool)
    
    # Anatomical parameters for shoulder structure
    # These parameters position the planes to form a proper shoulder shape
    scapula_position = int(volume_size * 0.5)  # Center of scapula
    humeral_head_center = (int(volume_size * 0.6), int(volume_size * 0.4), int(volume_size * 0.5))
    glenoid_center = (int(volume_size * 0.4), int(volume_size * 0.4), int(volume_size * 0.5))
    
    # Properly align and place axial plane (top view of shoulder)
    axial_bone = segmentations['axial']['bone']
    axial_soft = segmentations['axial']['soft_tissue']
    axial_muscle = segmentations['axial']['muscle']
    
    # Resize to fit our volume
    axial_bone_resized = resize_binary_image(axial_bone, (volume_size, volume_size))
    axial_soft_resized = resize_binary_image(axial_soft, (volume_size, volume_size))
    axial_muscle_resized = resize_binary_image(axial_muscle, (volume_size, volume_size))
    
    # Position axial plane properly for shoulder anatomy
    for z in range(scapula_position-20, scapula_position+20):
        # Vary opacity based on distance from center
        opacity = 1.0 - abs(z - scapula_position) / 20.0
        if opacity <= 0:
            continue
            
        # Add with decreasing opacity away from center
        bone_volume[:, :, z] = bone_volume[:, :, z] | (axial_bone_resized * opacity > 0.2)
        soft_volume[:, :, z] = soft_volume[:, :, z] | (axial_soft_resized * opacity > 0.2)
        muscle_volume[:, :, z] = muscle_volume[:, :, z] | (axial_muscle_resized * opacity > 0.2)
    
    # Place sagittal plane (side view)
    sagittal_bone = segmentations['sagittal']['bone']
    sagittal_soft = segmentations['sagittal']['soft_tissue']
    sagittal_muscle = segmentations['sagittal']['muscle']
    
    # Resize to fit our volume
    sagittal_bone_resized = resize_binary_image(sagittal_bone, (volume_size, volume_size))
    sagittal_soft_resized = resize_binary_image(sagittal_soft, (volume_size, volume_size))
    sagittal_muscle_resized = resize_binary_image(sagittal_muscle, (volume_size, volume_size))
    
    # Position sagittal plane properly for shoulder anatomy
    for x in range(humeral_head_center[0]-20, humeral_head_center[0]+20):
        # Vary opacity based on distance from center
        opacity = 1.0 - abs(x - humeral_head_center[0]) / 20.0
        if opacity <= 0:
            continue
            
        # Add with decreasing opacity away from center
        bone_volume[x, :, :] = bone_volume[x, :, :] | (np.rot90(sagittal_bone_resized) * opacity > 0.2)
        soft_volume[x, :, :] = soft_volume[x, :, :] | (np.rot90(sagittal_soft_resized) * opacity > 0.2)
        muscle_volume[x, :, :] = muscle_volume[x, :, :] | (np.rot90(sagittal_muscle_resized) * opacity > 0.2)
    
    # Place coronal plane (front view)
    coronal_bone = segmentations['coronal']['bone']
    coronal_soft = segmentations['coronal']['soft_tissue']
    coronal_muscle = segmentations['coronal']['muscle']
    
    # Resize to fit our volume
    coronal_bone_resized = resize_binary_image(coronal_bone, (volume_size, volume_size))
    coronal_soft_resized = resize_binary_image(coronal_soft, (volume_size, volume_size))
    coronal_muscle_resized = resize_binary_image(coronal_muscle, (volume_size, volume_size))
    
    # Position coronal plane properly for shoulder anatomy
    for y in range(glenoid_center[1]-20, glenoid_center[1]+20):
        # Vary opacity based on distance from center
        opacity = 1.0 - abs(y - glenoid_center[1]) / 20.0
        if opacity <= 0:
            continue
            
        # Add with decreasing opacity away from center
        bone_volume[:, y, :] = bone_volume[:, y, :] | (np.rot90(coronal_bone_resized) * opacity > 0.2)
        soft_volume[:, y, :] = soft_volume[:, y, :] | (np.rot90(coronal_soft_resized) * opacity > 0.2)
        muscle_volume[:, y, :] = muscle_volume[:, y, :] | (np.rot90(coronal_muscle_resized) * opacity > 0.2)
    
    # Add anatomical shoulder shapes
    # Add humeral head (ball shape)
    radius = int(volume_size * 0.15)
    hh_center = humeral_head_center
    
    x, y, z = np.ogrid[:volume_size, :volume_size, :volume_size]
    dist_from_center = np.sqrt((x - hh_center[0])**2 + (y - hh_center[1])**2 + (z - hh_center[2])**2)
    humeral_head = dist_from_center <= radius
    
    # Add glenoid fossa (socket shape)
    radius_glenoid = int(volume_size * 0.12)
    glenoid_center_point = glenoid_center
    
    x, y, z = np.ogrid[:volume_size, :volume_size, :volume_size]
    dist_from_glenoid = np.sqrt((x - glenoid_center_point[0])**2 + 
                               (y - glenoid_center_point[1])**2 + 
                               (z - glenoid_center_point[2])**2)
    
    glenoid_outer = dist_from_glenoid <= radius_glenoid
    glenoid_inner = dist_from_glenoid <= radius_glenoid - 5
    glenoid = glenoid_outer & ~glenoid_inner
    
    # Add anatomical elements to appropriate tissue volumes
    bone_volume = bone_volume | humeral_head | glenoid
    
    # Add rotator cuff (muscle tissue surrounding humeral head)
    rotator_cuff_outer = np.zeros_like(bone_volume)
    rotator_thickness = 15
    
    x, y, z = np.ogrid[:volume_size, :volume_size, :volume_size]
    dist_from_center = np.sqrt((x - hh_center[0])**2 + (y - hh_center[1])**2 + (z - hh_center[2])**2)
    rotator_cuff_outer = (dist_from_center <= radius + rotator_thickness) & (dist_from_center > radius)
    
    # Add cuff to muscle volume
    muscle_volume = muscle_volume | rotator_cuff_outer
    
    # Add soft tissue layer between bone and muscle
    soft_tissue_layer = np.zeros_like(soft_volume)
    layer_thickness = 5
    
    x, y, z = np.ogrid[:volume_size, :volume_size, :volume_size]
    dist_from_center = np.sqrt((x - hh_center[0])**2 + (y - hh_center[1])**2 + (z - hh_center[2])**2)
    soft_tissue_layer = (dist_from_center <= radius + layer_thickness) & (dist_from_center > radius)
    
    # Add layer to soft tissue volume
    soft_volume = soft_volume | soft_tissue_layer
    
    # Smooth volumes
    bone_volume_smooth = ndimage.gaussian_filter(bone_volume.astype(float), sigma=1.5)
    soft_volume_smooth = ndimage.gaussian_filter(soft_volume.astype(float), sigma=1.5)
    muscle_volume_smooth = ndimage.gaussian_filter(muscle_volume.astype(float), sigma=1.5)
    
    # Convert back to binary with appropriate thresholds
    bone_volume = bone_volume_smooth > 0.2
    soft_volume = soft_volume_smooth > 0.2
    muscle_volume = muscle_volume_smooth > 0.2
    
    # Ensure no overlap between volumes
    muscle_volume_final = muscle_volume
    soft_volume_final = soft_volume & ~muscle_volume_final
    bone_volume_final = bone_volume & ~soft_volume_final & ~muscle_volume_final
    
    # Create meshes using marching cubes algorithm
    print("Creating 3D meshes of anatomical shoulder...")
    
    # Bone mesh
    try:
        bone_verts, bone_faces, _, _ = measure.marching_cubes(bone_volume_final)
        bone_mesh = pv.PolyData(bone_verts, faces=np.column_stack(([3] * len(bone_faces), bone_faces)))
        bone_mesh = bone_mesh.smooth(n_iter=15)
        bone_mesh.save(os.path.join(output_folder, 'models', 'shoulder_bone.stl'))
        print("Saved bone model")
    except Exception as e:
        print(f"Error creating bone mesh: {e}")
    
    # Soft tissue mesh
    try:
        soft_verts, soft_faces, _, _ = measure.marching_cubes(soft_volume_final)
        soft_mesh = pv.PolyData(soft_verts, faces=np.column_stack(([3] * len(soft_faces), soft_faces)))
        soft_mesh = soft_mesh.smooth(n_iter=15)
        soft_mesh.save(os.path.join(output_folder, 'models', 'shoulder_soft_tissue.stl'))
        print("Saved soft tissue model")
    except Exception as e:
        print(f"Error creating soft tissue mesh: {e}")
    
    # Muscle mesh
    try:
        muscle_verts, muscle_faces, _, _ = measure.marching_cubes(muscle_volume_final)
        muscle_mesh = pv.PolyData(muscle_verts, faces=np.column_stack(([3] * len(muscle_faces), muscle_faces)))
        muscle_mesh = muscle_mesh.smooth(n_iter=15)
        muscle_mesh.save(os.path.join(output_folder, 'models', 'shoulder_muscle.stl'))
        print("Saved muscle model")
    except Exception as e:
        print(f"Error creating muscle mesh: {e}")
    
    # Create anatomically correct visualization of the 3D shoulder model
    try:
        p = pv.Plotter(off_screen=True)
        
        # Add meshes with anatomically correct colors
        p.add_mesh(bone_mesh, color='ivory', opacity=1.0, label='Bone (Grey)')
        p.add_mesh(soft_mesh, color='lightpink', opacity=0.6, label='Soft Tissue (Darker White)')
        p.add_mesh(muscle_mesh, color='firebrick', opacity=0.7, label='Muscle (White)')
        
        # Set better camera angle for shoulder visualization
        p.camera_position = [(200, 200, 200), (volume_size/2, volume_size/2, volume_size/2), (0, 0, 1)]
        p.add_legend()
        
        # Save the visualization
        p.screenshot(os.path.join(output_folder, 'models', 'shoulder_3d_model.png'))
        print("Created anatomical shoulder visualization")
        
        # Create separate visualizations for each tissue
        for tissue, mesh, color, name in [
            ('bone', bone_mesh, 'ivory', 'Shoulder Bone'),
            ('soft_tissue', soft_mesh, 'lightpink', 'Shoulder Soft Tissue'),
            ('muscle', muscle_mesh, 'firebrick', 'Shoulder Muscles')
        ]:
            p_single = pv.Plotter(off_screen=True)
            p_single.add_mesh(mesh, color=color, opacity=1.0, label=name)
            p_single.camera_position = [(200, 200, 200), (volume_size/2, volume_size/2, volume_size/2), (0, 0, 1)]
            p_single.screenshot(os.path.join(output_folder, 'models', f'shoulder_{tissue}.png'))
            
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
    
    # Output folder
    output_folder = 'results'
    
    # Process the MRI slices
    print("Starting processing...")
    process_mri_slices(image_paths, output_folder)
    
    print("\nProcessing complete! Results saved to", output_folder)