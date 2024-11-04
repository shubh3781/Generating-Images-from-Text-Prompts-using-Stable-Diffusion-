# Text-to-Image Generation with Stable Diffusion

This project demonstrates how to use the Stable Diffusion model to generate images from textual prompts. By leveraging the `diffusers` library from Hugging Face, this project provides a straightforward approach to transforming text into images using a pre-trained Stable Diffusion model.

## Project Overview

The objective of this project is to generate images based on natural language prompts. Stable Diffusion, a powerful text-to-image model, is employed here to create high-quality visuals directly from simple text descriptions.

## Features

- Generate images from text prompts
- Save generated images as PNG files
- Easy-to-use code structure with a dedicated function for image generation

## Installation

To get started with this project, ensure you have Python installed along with the necessary dependencies. Install them as follows:

```bash
pip install diffusers torch pillow


## Usage

1. **Clone the Repository**: Clone this project repository to your local machine.
2. **Set Up Model and Device**: The code initializes the model based on your available hardware (GPU if available, otherwise CPU).
3. **Generate Images**: Use the `generate_image` function to generate images based on custom text prompts.

### Code Example

```python
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Define model ID and device setup
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Define the image generation function
def generate_image(prompt):
    # Generate the image from the prompt
    with torch.no_grad():
        image = pipe(prompt).images[0]
    return image

# Test the function with a sample prompt
prompt = "sunset on beach"
image = generate_image(prompt)
image.save("sunset_on_beach.png")
```

## Example Prompt and Output

- **Prompt**: `"sunset on beach"`
- **Output**: The generated image will reflect the scene described by the prompt, with visual elements of a beach sunset.

## Requirements

- Python 3.7 or higher
- Required Python Libraries:
  - `diffusers`
  - `torch`
  - `Pillow`

## Notes

- Ensure your machine has a GPU for faster performance, though the code will work with a CPU as well.
- Model weights for Stable Diffusion may require significant storage, so ensure adequate space is available.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for the `diffusers` library
- Stability AI for the Stable Diffusion model

```