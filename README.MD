# Morphological Image Processing

This project demonstrates the implementation and analysis of morphological operations such as dilation, erosion, opening, and closing on various images.

## Objective

- To learn the operations of dilation and erosion.
- To learn the operations of opening and closing.
- To understand the characteristics of related algorithms.

## Requirements

1. Implement the related morphology algorithms and analyze the characteristics based on the given five images.
2. Finish the experimental report.

## Principles of the Related Algorithm

- **Dilation:** Enlarges the boundaries of objects in an image.
- **Erosion:** Shrinks the boundaries of objects in an image.
- **Opening:** Removes small objects from the foreground, typically noise.
- **Closing:** Fills small holes in the foreground and connects disjoint objects.

## Functions Overview

### MorphologicalOperations Class

- **__init__(kernel_size=(15, 15)):** Initializes with a kernel size.
- **dilate(image, iterations=3):** Applies dilation to an image.
- **erode(image, iterations=3):** Applies erosion to an image.
- **open(image, iterations=3):** Applies opening (erosion followed by dilation).
- **close(image, iterations=3):** Applies closing (dilation followed by erosion).

### ImageProcessor Class

- **__init__(filepaths):** Initializes with a list of image file paths.
- **apply_operations():** Applies dilation, erosion, opening, and closing to the images.
- **adjust_contrast_and_brightness(image, contrast=1.5, brightness=50):** Adjusts contrast and brightness.
- **overlay_difference(original, processed):** Overlays the difference between the original and processed images.
- **add_border_and_label(image, label, color=(255, 0, 0)):** Adds a border and label to the image.
- **display_combined_images(title, original_images, processed_images, operation):** Displays combined images with labels.
- **show_results():** Shows the results of the morphological operations.
- **save_images(output_dir):** Saves the processed images to a specified directory.
- **save_combined_images(output_dir):** Saves combined images (original and processed) to a specified directory.


## Original Images

Here are the original images used for the experiment:

<div style="display: flex; flex-wrap: wrap;">
  <img src="data/dilation.jpg" alt="Dilation Image" width="377" style="margin-right: 10px;">
  <img src="data/erosion.jpg" alt="Erosion Image" width="335" style="margin-right: 10px;">
<img src="data/opening.jpg" alt="Opening Image" width="745" style="margin-right: 10px;">
  <img src="data/erosion1.jpg" alt="Erosion1 Image" width="377" style="margin-right: 10px;">
  <img src="data/closing.jpg" alt="Closing Image" width="745">
</div>

## Results

### Dilation Results
![Dilation Results](output/dilation_combined.png)
![Dilation Results](output/combined/dilation.png)

### Erosion Results
![Erosion Results](output/erosion_combined.png)
![Erosion Results](output/combined/erosion.png)

### Opening Results
![Opening Results](output/opening_combined.png)
![Opening Results](output/combined/opening.png)

### Closing Results
![Closing Results](output/closing_combined.png)
![Closing Results](output/combined/closing.png)

## Conclusion

This project successfully implemented and analyzed the morphological operations of dilation, erosion, opening, and closing. By applying these operations to various images, their distinct effects were observed and visualized, providing a comprehensive understanding of their characteristics and practical applications in image processing.

## Repository

[GitHub Repository](https://github.com/Hetawk/morphological_image_processing.git)
