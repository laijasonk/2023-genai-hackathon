#!/usr/bin/env python3

"""Example of running Stable Diffusion locally.

You must be authenticated with gcloud before running.
$ gcloud auth application-default login
$ gcloud config set project gen-hybrid-intelligence-team-1
$ gcloud auth application-default set-quota-project gen-hybrid-intelligence-team-1
    
"""

import os
import sys

# AI imports 
from google.cloud import aiplatform
from keras_cv.models import StableDiffusion

# Image imports
from PIL import Image
import cv2

# Project constants
PROJECT_ID = "gen-hybrid-intelligence-team-1"
BUCKET_URI = ""
REGION = "us-central1"
STAGING_BUCKET = os.path.join(BUCKET_URI, "temporal")
EXPERIMENT_BUCKET = os.path.join(BUCKET_URI, "keras")
DATA_BUCKET = os.path.join(EXPERIMENT_BUCKET, "data")
MODEL_BUCKET = os.path.join(EXPERIMENT_BUCKET, "model")

# Output constants
RESOLUTION = 256

# Initiate VertexAI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=STAGING_BUCKET)

# Create model
model_path = "" # Empty path uses default Keras model
model = StableDiffusion(img_height=RESOLUTION, img_width=RESOLUTION, jit_compile=True)
if model_path:
    model.diffusion_model.load_weights(model_path)

# Create
batch_size = 1
img = model.text_to_image(
    #prompt="headless mannequin display with a black dress and long-sleeves",
    prompt="female model wearing a black dress and long-sleeves",
    batch_size=batch_size,
    num_steps=25,
    seed=42,
)

for idx in range(batch_size):
    Image.fromarray(img[idx]).save(f"./image_{idx}.jpg")

