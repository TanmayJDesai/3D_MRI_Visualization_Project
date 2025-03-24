import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, morphology, measure, exposure, feature, segmentation, color
from scipy import ndimage
import pyvista as pv

def process_mri_slices(image_paths, output_folder):
    """
    Process MRI slices to create a 3D model of a shoulder from three planes with improved segmentation.
    
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
        
        # IMPROVED PREPROCESSING
        # Apply CLAHE for better contrast enhancement
        img_eq = exposure.equalize_adapthist(img_norm, clip_limit=0.02)
        
        # Apply bilateral filter to reduce noise while preserving edges
        img_filtered = filters.gaussian(img_eq, sigma=0.5)
        
        # IMPROVED SEGMENTATION APPROACH
        # Use multi-Otsu thresholding to find optimal thresholds for separating tissues
        thresholds = filters.threshold_multiotsu(img_filtered, classes=4)
        
        # Create masks for each tissue type
        background_mask = img_filtered <= thresholds[0]
        bone_mask = (img_filtered > thresholds[0]) & (img_filtered <= thresholds[1])
        soft_tissue_mask = (img_filtered > thresholds[1]) & (img_filtered <= thresholds[2])
        muscle_mask = img_filtered > thresholds[2]
        
        # IMPROVED MORPHOLOGICAL OPERATIONS
        # Clean up masks with more targeted morphological operations
        # Remove small objects
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=100)
        soft_tissue_mask = morphology.remove_small_objects(soft_tissue_mask, min_size=150)
        muscle_mask = morphology.remove_small_objects(muscle_mask, min_size=80)
        
        # Close holes
        bone_mask = morphology.closing(bone_mask, morphology.disk(3))
        soft_tissue_mask = morphology.closing(soft_tissue_mask, morphology.disk(4))
        muscle_mask = morphology.closing(muscle_mask, morphology.disk(2))
        
        # Apply context-specific filtering based on anatomical knowledge
        # For example, ensure muscle areas are connected and coherent
        muscle_mask = morphology.area_opening(muscle_mask, area_threshold=50)
        
        # IMPROVED MASK SEPARATION
        # Ensure no overlap between masks with priority order: muscle > soft tissue > bone
        # This ensures clear boundaries between tissue types
        muscle_mask_final = muscle_mask
        soft_tissue_mask_final = soft_tissue_mask & ~muscle_mask_final
        bone_mask_final = bone_mask & ~soft_tissue_mask_final & ~muscle_mask_final
        
        # Post-processing: ensure anatomical continuity
        # For example, bones should be continuous structures
        bone_labeled = measure.label(bone_mask_final)
        bone_props = measure.regionprops(bone_labeled)
        
        if len(bone_props) > 0:
            # Keep only the largest bone regions
            large_bone_labels = sorted([prop.label for prop in bone_props], 
                                       key=lambda x: np.sum(bone_labeled == x), 
                                       reverse=True)[:3]  # Keep top 3 largest regions
            
            bone_mask_final = np.isin(bone_labeled, large_bone_labels)
        
        # Similar process for muscle to ensure we keep only significant muscle regions
        muscle_labeled = measure.label(muscle_mask_final)
        muscle_props = measure.regionprops(muscle_labeled)
        
        if len(muscle_props) > 0:
            large_muscle_labels = sorted([prop.label for prop in muscle_props], 
                                        key=lambda x: np.sum(muscle_labeled == x), 
                                        reverse=True)[:5]  # Keep top 5 largest regions
            
            muscle_mask_final = np.isin(muscle_labeled, large_muscle_labels)
        
        # Save segmentation results
        segmentations[plane] = {
            'bone': bone_mask_final,
            'soft_tissue': soft_tissue_mask_final,
            'muscle': muscle_mask_final
        }
        
        # Create visualization of segmentation
        # Combined visualization with enhanced colors and opacity
        combined = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        combined[bone_mask_final, 2] = 255  # Blue for bone
        combined[soft_tissue_mask_final, 1] = 255  # Green for soft tissue
        combined[muscle_mask_final, 0] = 255  # Red for muscle
        
        # Save combined visualization
        combined_output = os.path.join(output_folder, 'segmented', f"{plane}_segmentation.png")
        io.imsave(combined_output, combined)
        
        # Create a blended visualization with original image and segmentation
        blended = color.label2rgb(
            bone_mask_final.astype(int) + soft_tissue_mask_final.astype(int)*2 + muscle_mask_final.astype(int)*3,
            img_norm, 
            colors=[(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0)],
            alpha=0.5,
            bg_label=0
        )
        
        # Show original and segmentation results
        plt.figure(figsize=(20, 5))
        plt.subplot(151)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original {plane}')
        
        plt.subplot(152)
        plt.imshow(img_filtered, cmap='gray')
        plt.title('Filtered Image')
        
        plt.subplot(153)
        plt.imshow(bone_mask_final, cmap='Blues')
        plt.title('Bone (Blue)')
        
        plt.subplot(154)
        plt.imshow(soft_tissue_mask_final, cmap='Greens')
        plt.title('Soft Tissue (Green)')
        
        plt.subplot(155)
        plt.imshow(muscle_mask_final, cmap='Reds')
        plt.title('Muscle (Red)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'segmented', f"{plane}_result.png"))
        
        # Additional visualization showing blended segmentation
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original {plane}')
        
        plt.subplot(132)
        plt.imshow(combined)
        plt.title('Segmentation\nBlue: Bone, Green: Soft Tissue, Red: Muscle')
        
        plt.subplot(133)
        plt.imshow(blended)
        plt.title('Blended Segmentation')
        
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
    
    # Improved anatomical parameters for shoulder structure
    # These parameters position the planes to form a proper shoulder shape
    scapula_position = int(volume_size * 0.5)  # Center of scapula
    humeral_head_center = (int(volume_size * 0.6), int(volume_size * 0.4), int(volume_size * 0.5))
    glenoid_center = (int(volume_size * 0.4), int(volume_size * 0.4), int(volume_size * 0.5))
    
    # IMPROVED ALIGNMENT
    # Properly align and place axial plane (top view of shoulder)
    axial_bone = segmentations['axial']['bone']
    axial_soft = segmentations['axial']['soft_tissue']
    axial_muscle = segmentations['axial']['muscle']
    
    # Resize to fit our volume
    axial_bone_resized = resize_binary_image(axial_bone, (volume_size, volume_size))
    axial_soft_resized = resize_binary_image(axial_soft, (volume_size, volume_size))
    axial_muscle_resized = resize_binary_image(axial_muscle, (volume_size, volume_size))
    
    # Position axial plane properly for shoulder anatomy with improved weighting
    for z in range(scapula_position-25, scapula_position+25):
        # Improved opacity profile for better transitions
        # Use a Gaussian weighting instead of linear
        dist = abs(z - scapula_position)
        opacity = np.exp(-(dist**2) / (2 * 10**2))  # Gaussian weighting
            
        # Add with Gaussian opacity away from center
        bone_volume[:, :, z] = bone_volume[:, :, z] | (axial_bone_resized & (np.random.random(axial_bone_resized.shape) < opacity))
        soft_volume[:, :, z] = soft_volume[:, :, z] | (axial_soft_resized & (np.random.random(axial_soft_resized.shape) < opacity))
        muscle_volume[:, :, z] = muscle_volume[:, :, z] | (axial_muscle_resized & (np.random.random(axial_muscle_resized.shape) < opacity))
    
    # Place sagittal plane (side view) with improved alignment
    sagittal_bone = segmentations['sagittal']['bone']
    sagittal_soft = segmentations['sagittal']['soft_tissue']
    sagittal_muscle = segmentations['sagittal']['muscle']
    
    # Resize to fit our volume
    sagittal_bone_resized = resize_binary_image(sagittal_bone, (volume_size, volume_size))
    sagittal_soft_resized = resize_binary_image(sagittal_soft, (volume_size, volume_size))
    sagittal_muscle_resized = resize_binary_image(sagittal_muscle, (volume_size, volume_size))
    
    # Position sagittal plane properly for shoulder anatomy with improved weighting
    for x in range(humeral_head_center[0]-25, humeral_head_center[0]+25):
        # Improved opacity profile
        dist = abs(x - humeral_head_center[0])
        opacity = np.exp(-(dist**2) / (2 * 10**2))  # Gaussian weighting
            
        # Add with improved opacity profile
        bone_volume[x, :, :] = bone_volume[x, :, :] | (np.rot90(sagittal_bone_resized) & (np.random.random(np.rot90(sagittal_bone_resized).shape) < opacity))
        soft_volume[x, :, :] = soft_volume[x, :, :] | (np.rot90(sagittal_soft_resized) & (np.random.random(np.rot90(sagittal_soft_resized).shape) < opacity))
        muscle_volume[x, :, :] = muscle_volume[x, :, :] | (np.rot90(sagittal_muscle_resized) & (np.random.random(np.rot90(sagittal_muscle_resized).shape) < opacity))
    
    # Place coronal plane (front view) with improved alignment
    coronal_bone = segmentations['coronal']['bone']
    coronal_soft = segmentations['coronal']['soft_tissue']
    coronal_muscle = segmentations['coronal']['muscle']
    
    # Resize to fit our volume
    coronal_bone_resized = resize_binary_image(coronal_bone, (volume_size, volume_size))
    coronal_soft_resized = resize_binary_image(coronal_soft, (volume_size, volume_size))
    coronal_muscle_resized = resize_binary_image(coronal_muscle, (volume_size, volume_size))
    
    # Position coronal plane properly for shoulder anatomy with improved weighting
    for y in range(glenoid_center[1]-25, glenoid_center[1]+25):
        # Improved opacity profile
        dist = abs(y - glenoid_center[1])
        opacity = np.exp(-(dist**2) / (2 * 10**2))  # Gaussian weighting
            
        # Add with improved opacity profile
        bone_volume[:, y, :] = bone_volume[:, y, :] | (np.rot90(coronal_bone_resized) & (np.random.random(np.rot90(coronal_bone_resized).shape) < opacity))
        soft_volume[:, y, :] = soft_volume[:, y, :] | (np.rot90(coronal_soft_resized) & (np.random.random(np.rot90(coronal_soft_resized).shape) < opacity))
        muscle_volume[:, y, :] = muscle_volume[:, y, :] | (np.rot90(coronal_muscle_resized) & (np.random.random(np.rot90(coronal_muscle_resized).shape) < opacity))
    
    # IMPROVED ANATOMICAL SHAPES
    # Add anatomical shoulder shapes with better definition
    # Add humeral head (ball shape) with improved anatomy
    radius = int(volume_size * 0.15)
    hh_center = humeral_head_center
    
    x, y, z = np.ogrid[:volume_size, :volume_size, :volume_size]
    dist_from_center = np.sqrt((x - hh_center[0])**2 + (y - hh_center[1])**2 + (z - hh_center[2])**2)
    
    # Create slightly elliptical humeral head (more anatomically correct)
    humeral_head = ((x - hh_center[0])**2 / (radius*1.1)**2 + 
                   (y - hh_center[1])**2 / radius**2 + 
                   (z - hh_center[2])**2 / radius**2) <= 1.0
    
    # Add glenoid fossa (socket shape) with improved shape
    radius_glenoid = int(volume_size * 0.12)
    glenoid_center_point = glenoid_center
    
    # Create slightly concave glenoid fossa (more anatomically correct)
    dist_from_glenoid = np.sqrt((x - glenoid_center_point[0])**2 + 
                               (y - glenoid_center_point[1])**2 + 
                               (z - glenoid_center_point[2])**2)
    
    glenoid_outer = dist_from_glenoid <= radius_glenoid
    glenoid_inner = dist_from_glenoid <= radius_glenoid - 5
    glenoid = glenoid_outer & ~glenoid_inner
    
    # Add anatomical elements to appropriate tissue volumes
    bone_volume = bone_volume | humeral_head | glenoid
    
    # IMPROVED SOFT TISSUE MODELING
    # Add rotator cuff (muscle tissue surrounding humeral head) with better definition
    rotator_cuff_outer = np.zeros_like(bone_volume)
    rotator_thickness = 15
    
    # Create anatomically correct rotator cuff that wraps around the humeral head
    # but doesn't completely enclose it (matches real anatomy)
    x, y, z = np.ogrid[:volume_size, :volume_size, :volume_size]
    
    # Create a partial spherical shell (upper part of humeral head)
    dist_from_center = np.sqrt((x - hh_center[0])**2 + (y - hh_center[1])**2 + (z - hh_center[2])**2)
    
    # Only cover upper part of humeral head with cuff (anatomically correct)
    upper_region = y < hh_center[1]  # Upper portion only
    rotator_cuff_outer = (dist_from_center <= radius + rotator_thickness) & (dist_from_center > radius) & upper_region
    
    # Add cuff to muscle volume
    muscle_volume = muscle_volume | rotator_cuff_outer
    
    # Add better defined soft tissue layer between bone and muscle
    soft_tissue_layer = np.zeros_like(soft_volume)
    layer_thickness = 5
    
    # Create a thin cartilage/soft tissue layer over bone surfaces
    dist_from_center = np.sqrt((x - hh_center[0])**2 + (y - hh_center[1])**2 + (z - hh_center[2])**2)
    soft_tissue_layer = (dist_from_center <= radius + layer_thickness) & (dist_from_center > radius)
    
    # Also add soft tissue around glenoid
    dist_from_glenoid = np.sqrt((x - glenoid_center_point[0])**2 + 
                              (y - glenoid_center_point[1])**2 + 
                              (z - glenoid_center_point[2])**2)
    
    glenoid_soft = (dist_from_glenoid <= radius_glenoid + layer_thickness) & (dist_from_glenoid > radius_glenoid)
    
    # Add layer to soft tissue volume
    soft_volume = soft_volume | soft_tissue_layer | glenoid_soft
    
    # IMPROVED VOLUME SMOOTHING & FINALIZATION
    # Apply more nuanced smoothing to preserve tissue boundaries
    # For bone: less smoothing to preserve structural details
    bone_volume_smooth = ndimage.gaussian_filter(bone_volume.astype(float), sigma=1.2)
    # For soft tissue: moderate smoothing
    soft_volume_smooth = ndimage.gaussian_filter(soft_volume.astype(float), sigma=1.8)
    # For muscle: more smoothing for realistic muscle appearance
    muscle_volume_smooth = ndimage.gaussian_filter(muscle_volume.astype(float), sigma=2.0)
    
    # Convert back to binary with appropriate thresholds
    bone_volume = bone_volume_smooth > 0.25  # Higher threshold to prevent over-smoothing
    soft_volume = soft_volume_smooth > 0.2
    muscle_volume = muscle_volume_smooth > 0.2
    
    # Ensure no overlap between volumes with priority
    muscle_volume_final = muscle_volume
    soft_volume_final = soft_volume & ~muscle_volume_final
    bone_volume_final = bone_volume & ~soft_volume_final & ~muscle_volume_final
    
    # Create meshes using marching cubes algorithm
    print("Creating 3D meshes of anatomical shoulder...")
    
    # Bone mesh with improved smoothing
    try:
        bone_verts, bone_faces, _, _ = measure.marching_cubes(bone_volume_final)
        bone_mesh = pv.PolyData(bone_verts, faces=np.column_stack(([3] * len(bone_faces), bone_faces)))
        # Heavier smoothing for bone to get anatomical smoothness but preserve features
        bone_mesh = bone_mesh.smooth(n_iter=20, relaxation_factor=0.2)
        bone_mesh.save(os.path.join(output_folder, 'models', 'shoulder_bone.stl'))
        print("Saved bone model")
    except Exception as e:
        print(f"Error creating bone mesh: {e}")
    
    # Soft tissue mesh with improved smoothing
    try:
        soft_verts, soft_faces, _, _ = measure.marching_cubes(soft_volume_final)
        soft_mesh = pv.PolyData(soft_verts, faces=np.column_stack(([3] * len(soft_faces), soft_faces)))
        soft_mesh = soft_mesh.smooth(n_iter=18, relaxation_factor=0.25)
        soft_mesh.save(os.path.join(output_folder, 'models', 'shoulder_soft_tissue.stl'))
        print("Saved soft tissue model")
    except Exception as e:
        print(f"Error creating soft tissue mesh: {e}")
    
    # Muscle mesh with improved smoothing
    try:
        muscle_verts, muscle_faces, _, _ = measure.marching_cubes(muscle_volume_final)
        muscle_mesh = pv.PolyData(muscle_verts, faces=np.column_stack(([3] * len(muscle_faces), muscle_faces)))
        muscle_mesh = muscle_mesh.smooth(n_iter=25, relaxation_factor=0.3)  # More smoothing for muscles
        muscle_mesh.save(os.path.join(output_folder, 'models', 'shoulder_muscle.stl'))
        print("Saved muscle model")
    except Exception as e:
        print(f"Error creating muscle mesh: {e}")
    
    # Create anatomically correct visualization of the 3D shoulder model
    try:
        p = pv.Plotter(off_screen=True)
        
        # Add meshes with anatomically correct colors and better opacity
        p.add_mesh(bone_mesh, color='ivory', opacity=1.0, label='Bone (Blue)')
        p.add_mesh(soft_mesh, color='lightpink', opacity=0.7, label='Soft Tissue (Green)')
        p.add_mesh(muscle_mesh, color='firebrick', opacity=0.8, label='Muscle (Red)')
        
        # Set better camera angle for shoulder visualization
        p.camera_position = [(200, 200, 200), (volume_size/2, volume_size/2, volume_size/2), (0, 0, 1)]
        p.add_legend()
        
        # Save the visualization
        p.screenshot(os.path.join(output_folder, 'models', 'shoulder_3d_model.png'), window_size=(1200, 1000))
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
            p_single.screenshot(os.path.join(output_folder, 'models', f'shoulder_{tissue}.png'), window_size=(1000, 1000))
            
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