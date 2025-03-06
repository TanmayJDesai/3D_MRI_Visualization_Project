# MRI Slice Segmentation & 3D Reconstruction

## Overview
This project processes **MRI slice images** from three different planes (**axial, sagittal, coronal**) to **segment tissues (bone and soft tissue)** and generate a **3D reconstruction** of the anatomy.

Using **image processing, morphological operations, and 3D meshing techniques**, I segment important structures and create 3D models that can be visualized and exported.

## Libraries Used

### Image Processing & Segmentation
* `numpy` - For array manipulation and numerical operations
* `matplotlib.pyplot` - To visualize MRI slices and segmentation masks
* `skimage.io` - To load MRI images
* `skimage.filters` - For thresholding (Otsu's method) to segment bone and tissue
* `skimage.morphology` - For morphological operations like **closing** and **removing small objects**
* `skimage.measure` - To extract 3D surfaces using the **marching cubes** algorithm
* `scipy.ndimage` - To apply **binary dilation** and **Gaussian smoothing** for filling missing data
* `skimage.exposure` - To enhance contrast via histogram equalization

### 3D Modeling & Visualization
* `pyvista` - To generate, visualize, and export **3D models (.stl files)**

## How It Works

### 1. Load MRI Images
I load **three grayscale MRI slices** from different anatomical planes:
* **Axial (top-down view)**
* **Sagittal (side view)**
* **Coronal (front view)**

### 2. Image Enhancement
To improve contrast, I apply **histogram equalization** using `exposure.equalize_hist`.

### 3. Tissue Segmentation
I segment two key tissue types:
* **Bone**:
   * I use **Otsu's thresholding** to identify high-intensity regions
   * I apply **morphological operations** to clean up the segmentation mask
* **Soft Tissue**:
   * I define soft tissue based on **intensity ranges**
   * Similar **morphological operations** are applied to remove noise

### 4. Save Segmentation Results
For each MRI slice, I save:
* **Bone mask (Red)**
* **Soft tissue mask (Green)**
* **Overlay of both masks on the original image**

### 5. Create a 3D Volume
* I initialize empty **3D arrays** for bone and soft tissue
* I insert the **segmented 2D slices** into the corresponding positions in the **3D volume**
* I use **binary dilation** and **Gaussian smoothing** to fill in missing data between slices

### 6. Generate a 3D Model
* I apply the **marching cubes algorithm** to extract a **3D surface** from the segmented volume
* I create a **3D mesh** and export it as `.stl` files

### 7. Save & Display the 3D Models
* The **final 3D models** are stored in the `"models"` folder
* I use `pyvista` to **visualize and inspect** the 3D reconstructions

## Installation & Usage

### 1. Install Dependencies
Make sure you have Python installed, then install the required libraries:

```bash
pip install numpy matplotlib scikit-image scipy pyvista
```

### 2. Run the Script
Place your **MRI slices** in the input folder and run:

```bash
python main.py
```

### 3. View Results
* Segmented images are saved in `output/`
* 3D models are saved as `.stl` files in `models/`
* Use any 3D viewer (e.g., MeshLab, Blender) to inspect the **3D reconstructions**

## Future Improvements
* **More precise segmentation** using **machine learning**
* **Automated tissue classification** with **deep learning models**
* **Higher-quality 3D reconstructions** by incorporating **more MRI slices**

## Contributing
Feel free to fork the repository and submit pull requests for improvements!

## License
This project is open-source under the **MIT License**.
