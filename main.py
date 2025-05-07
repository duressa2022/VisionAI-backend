import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
app = FastAPI()

# Configure Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('models/gemini-1.5-pro')

# Define object schema
class ObjectData(BaseModel):
    label: str
    count: int
    confidence: float
    positions: List[Tuple[int, int]]

class VisionRequest(BaseModel):
    objects: List[ObjectData]
    timestamp: str


# Helper to craft prompt
def create_prompt(objects: List[ObjectData], timestamp: str) -> str:
    description = []
    for obj in objects:
        positions = ', '.join([f"({x}, {y})" for x, y in obj.positions])
        description.append(
            f"There are {obj.count} {obj.label}(s) detected with {obj.confidence*100:.1f}% confidence at positions {positions}."
        )

    scene_description = "\n".join(description)
    prompt = f"""
            You are an intelligent, kind, and creative narrator for a blind person.
            Based on object detection data from a camera, describe what is happening in a beautiful, gentle, and real-time spoken narration.

            Speak as if you're right beside the person, describing the world with care and imagination.
            Do not mention technical details like "confidence" or "bounding boxes".

            Focus on what the user might experience if they could see: people walking, objects nearby, interactions between them, etc.
            Use emotionally intelligent, sensory-rich, but natural language — like a narrator for a blind friend.

            Here is the object data detected at {timestamp}:

            {scene_description}

            Now, generate one concise and poetic narration in the present tense:
            """
    return prompt


# Endpoint
@app.post("/narrate")
async def narrate_scene(request: VisionRequest):
    try:
        prompt = create_prompt(request.objects, request.timestamp)
        response = model.generate_content(prompt)
        narration = response.text.strip()
        return {"narration": narration}
    except Exception as e:
        return {"error": str(e)}
