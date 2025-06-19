# Instagram Grid Layout Generator

A Python tool that helps you create aesthetically pleasing Instagram grid layouts based on the color profiles of your images.

## Features

- Analyzes the dominant colors in your photos
- Generates multiple layout options based on color harmony
- Creates visually pleasing 3x3 grid arrangements for your Instagram profile
- Provides options for maximum color harmony, color grouping, and gradient effects
- Visualizes and saves layout options for easy selection

## Requirements

- Python 3.7+
- Required packages:
  - numpy
  - pillow (PIL)
  - matplotlib
  - scikit-learn
  - colorsys (standard library)

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install numpy pillow matplotlib scikit-learn
```

## Usage

### Command Line Interface

Run the test script to use the program interactively:

```bash
python test_script.py
```

Follow the prompts to:
1. Enter the path to your image folder
2. Choose how many layout options you want
3. View layouts interactively and/or save them to disk

### Programmatic Usage

You can also use the `InstagramGridLayoutGenerator` class in your own code:

```python
from instagram_grid_layout import InstagramGridLayoutGenerator

# Initialize with your image folder
generator = InstagramGridLayoutGenerator("path/to/images")

# Load the images
generator.load_images()

# Extract dominant colors from each image
generator.extract_dominant_colors(num_colors=3)

# Generate layout options
layouts = generator.generate_layouts(num_layouts=5)

# Visualize a specific layout (0-indexed)
generator.visualize_layout(0)

# Save layout visualizations to disk
for i in range(len(layouts)):
    generator.save_layout_visualization("output_folder", i)
```

## Layout Options

The generator currently provides these layout strategies:

1. **Maximum Adjacent Harmony**: Arranges images to maximize color harmony between adjacent photos
2. **Color Grouping**: Groups images with similar color profiles
3. **Color Gradient**: Creates a gradient effect by arranging images by their average hue
4. **Random Arrangements**: Provides additional random layout options for variety

## Best Practices

- For optimal results, provide at least 9 images (for a standard Instagram grid)
- Images should be of similar dimensions for best visualization
- Try different layout options to find the one that best suits your aesthetic
- Consider the visual flow between images when selecting your layout

## Future Features

- Additional layout algorithms based on other visual properties
- Support for different grid sizes (e.g., 2x2, 4x4)
- Advanced color analysis options
- UI interface for easier interaction

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
