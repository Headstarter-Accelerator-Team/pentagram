from fastapi.responses import FileResponse
from flask import jsonify, send_file
from modal import Image, Secret, App, web_endpoint
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from gradio_client import Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Modal App
app = App("fastapi-gpu-api")

# Hugging Face token secret
hf_token = Secret.from_name("HF_TOKEN")

# Modal Image with dependencies
image = Image.debian_slim() \
    .pip_install_from_requirements("requirements.txt") \
    .pip_install("fastapi[standard]")

# Create FastAPI app
fastapi_app = FastAPI()

# Enable CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for saving images
save_directory = "/tmp/generated_images"
os.makedirs(save_directory, exist_ok=True)

# Request body model for image generation
class ImageRequest(BaseModel):
    textPart: str

# Route for generating images
@fastapi_app.post("/api/image_generator")
async def generate_image(request: ImageRequest):
    print("Received request body:", request)
    client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_token)

    prompt = request.textPart

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Generate the image using Hugging Face API
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )

        webp_image_path, _ = result
        save_path = os.path.join(save_directory, "generated_image.webp")
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove the existing file
        os.rename(webp_image_path, save_path)
        #file_name = request.args.get('file')
        #file_path = os.path.join(save_directory, file_name)

        # if not os.path.exists(file_path):
        #     return jsonify({'error': 'File not found'}), 404

        #return FileResponse(file_path, as_attachment=True)
        return FileResponse(
            path=save_path,
            media_type="image/webp",
            filename="generated_image.webp"
        )
        # Return the image URL for download
        return {"download_url": f"/tmp/generated_image.webp"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount FastAPI app as a Modal web endpoint
@app.function(gpu="A10G",image=image)
@web_endpoint(method="POST")
def run_fastapi_app(request: ImageRequest):
    # print("fast api app:")
    # return fastapi_app
    print("Received request body:", request)
    client = Client("black-forest-labs/FLUX.1-schnell", hf_token=hf_token)

    prompt = request.textPart

    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        # Generate the image using Hugging Face API
        result = client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=1024,
            height=1024,
            num_inference_steps=4,
            api_name="/infer"
        )

        webp_image_path, _ = result
        save_path = os.path.join(save_directory, "generated_image.webp")
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove the existing file
        os.rename(webp_image_path, save_path)
        #file_name = request.args.get('file')
        #file_path = os.path.join(save_directory, file_name)

        # if not os.path.exists(file_path):
        #     return jsonify({'error': 'File not found'}), 404

        #return FileResponse(file_path, as_attachment=True)
        return FileResponse(
            path=save_path,
            media_type="image/webp",
            filename="generated_image.webp"
        )
        # Return the image URL for download
        return {"download_url": f"/tmp/generated_image.webp"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Run locally for testing
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)