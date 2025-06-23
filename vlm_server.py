import base64
import io
from PIL import Image
import openai
from openai import OpenAI

# Initialize OpenAI client
key = ''
# Test values
testing_dict = {
    "yolo_boundary": [100, 150, 200, 250],  # [x_min, y_min, x_max, y_max]
    "confidence": 0.95,                     # Confidence score > 0.9
    "rgb_values": [120, 130, 140],         # Mean RGB values
    "depth_vector": [500, 510, 520, 530, 535, 550, 560, 570]  # Depth stats in mm
}

# Load and encode test image to base64
try:
    with open("/Users/AnirudhaDas/Downloads/testing_vlm/test_defect.png", "rb") as image_file:
        image = Image.open(image_file)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
except FileNotFoundError:
    print("Error: Image file not found. Please check the file path.")
    exit(1)
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Construct the prompt
prompt = f"""
Analyze the provided manufacturing image for defects. The image shows an object with the following stats:
- YOLO Boundary: {testing_dict['yolo_boundary']}
- Confidence: {testing_dict['confidence']}
- RGB Values: {testing_dict['rgb_values']}
- Depth Vector: {testing_dict['depth_vector']}

Identify the defect type (e.g., crack, misalignment) and provide a possible explanation for the defect based on the image and stats.
Return the response in JSON format with 'defect_type' and 'explanation' fields.
"""

try:
    # Send API request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ],
        max_tokens=300
    )
    
    # Print the response
    print(response.choices[0].message.content)

except openai.OpenAIError as e:
    print(f"OpenAI API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")