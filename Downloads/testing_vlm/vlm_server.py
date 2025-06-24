import base64
import io
from PIL import Image
import openai
from openai import OpenAI

def getreport(predmask_image, amap_image):


    # Initialize OpenAI client
    client = OpenAI()

    # Test values
    # testing_dict = {
    #     "yolo_boundary": [100, 150, 200, 250],  # [x_min, y_min, x_max, y_max]
    #     "confidence": 0.95,                     # Confidence score > 0.9
    #     "rgb_values": [120, 130, 140],         # Mean RGB values
    # }

    # Load and encode test image to base64
    try:
        with open(predmask_image, 'rb') as predmask_file:
            predmask = Image.open(predmask_file)
            buffered = io.BytesIO()
            predmask.save(buffered, format="PNG")
            predmask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        with open(amap_image, "rb") as amap_file:
            amap = Image.open(amap_file)
            buffered = io.BytesIO()
            amap.save(buffered, format="PNG")
            amap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    except FileNotFoundError:
        print("Error: Image file not found. Please check the file path.")
        exit(1)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)

    # Construct the prompts
    prompt = f"""
    Analyze the provided manufacturing images of an item for defects.
    2 images have been supplied for your convenience.
    Image #1 details a view of the object with the anomoly and a region of suspected defection.
    Image #2 details an anomaly depth map of the defected object.
    Identify the defect type (e.g., crack, misalignment) and provide a possible explanation for the defect based on the images.
    In the end, supply a one-sentence summary of the defect type and detailed concise explanation of the defect.
    Return the response in JSON format with 'defect_type', 'explanation', and 'one-liner' fields.
    """

    try:
        # Send API request
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{predmask_b64}"}},
                        {'type': 'image_url', "image_url": {"url": f"data:image/png;base64,{amap_b64}"}}
                    ]
                }
            ],
            max_tokens=200
        )
        
        # Print the response
        print(response.choices[0].message.content)

    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")