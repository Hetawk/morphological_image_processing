# main.py

from image_processor import ImageProcessor

if __name__ == "__main__":
    # File paths to the images
    filepaths = [
        'data/dilation.jpg',
        'data/erosion.jpg',
        'data/erosion1.jpg',
        'data/opening.jpg',
        'data/closing.jpg'
    ]

    # Initialize the image processor with the file paths
    processor = ImageProcessor(filepaths)

    # Apply morphological operations
    processor.apply_operations()

    # Display the results
    processor.show_results()

    # Save the results to a specified directory
    output_dir = 'output'
    processor.save_images(output_dir)
    processor.save_combined_images(output_dir)
