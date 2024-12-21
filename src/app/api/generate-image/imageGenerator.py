import modal
import io
from fastapi import Response, HTTPException, Query,Request
from datetime import datetime, timezone
import requests
import os

# Downloading our model
def downloadModel():
    from diffusers import AutoPipelineForText2Image
    import torch

    # Load the model with authentication token and other parameters
    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        use_auth_token=os.environ["HF_TOKEN"],
        torch_dtype=torch.float16,
        variant="fp16"
    )
    

# Setting up our image, specifing our image
# Spinning container and downloading these libraries into that container
image = (modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt") 
    .pip_install("fastapi[standard]", "transformers", "accelerate", "diffusers", "requests").run_function(downloadModel))

# Create our app, and we are passing the image with this app 
app = modal.App("sd_image_generator",image=image)

# Specificing our decorator to our class Model, setup the serverless env 
@app.cls(
    image=image,
    gpu = "A10G",
    container_idle_timeout=300, # Hitting the endpoint every 5 minutes to keep it warm
    secrets=[modal.Secret.from_name("API_KEY")] # Add this to make sure your endpoint is secure
)

class Model:
    # Define the function to load the model weight
    @modal.build() # Called when the modal app is building and packaging app to get ready for production
    @modal.enter()  # Needs to be called when the container is building and starts
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

        # Specify the model again to ensure it's loaded on the server
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        self.pipe.to("cuda") # Move the model to GPU
        self.API_KEY = ["API_KEY"] # This is within the docker container so we need to make sure it is secure here as well
    
     # Create Modal Endpoint, API endpoint that we query from next.js app backend
    @modal.web_endpoint()
    def generate(self,request: Request, prompt: str = Query(...,description="The prompt for image generation")):
        image = self.pipe(prompt, num_inference_steps=1,guidance_scale=0.0).images[0]

        api_key = request.header.get("X_API_KEY") # Get API Key from request header

        if api_key != self.API_KEY:
            raise Exception(
                status_code=401, 
                detail="HTTP: unauthorized access"
            )
        

 # At this point we have the image, now we need to load it into memory in a way we can return it a request
        buffer = io.BytesIO()  # Going to create a file that is in memory
        image.save(buffer, format="JPEG")

        return Response(content = buffer.getvalue(), media_type="image/jpeg")
    

     # Health endpoint to keep the container warm
    @modal.web_endpoint()
    def health(self):
        """Lightweight endpoint for eeping container warm"""
        return {"status":"healthy" , "timestamp": datetime.now(timezone.utc).isoformat()}

# At this point, the end result will be a URL
# Once we have the URL endpoint, we need to make sure we secure it with an API key

@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=(modal.Secret.from_name("API_KEY"))
)

def keep_warm(): 
    health_url= "deployed modal url-health" # Replace with actual health URL

    generate_url= "MODAL_ENDPOINT"  # Replace with actual generate URL

    #First check health endpoint
    health_response= requests.get(health_url)
    print(f"Health check at: {health_response.json() ['timestamp']}")

    #Then make a test request to generate endpoint with API key
    headers = {"X_API_KEY":os.environ["API-KEY"]}
    generate_response = requests.get(generate_url, headers=headers)
    print(f"Generate endpoint test successfully at: {datetime.now(timezone.utc).isoformat()}")
