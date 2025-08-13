from google.genai import Client
from google.genai.types import Part, GenerateContentConfig
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
from base64 import b64encode
import os
from dotenv import load_dotenv
from pydobot import Dobot
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_SYSTEM_INSTRUCTIONS = """
    You are a helpful assistant
"""


class DobotVision:
    """
    A class for integrating computer vision with a Dobot robotic arm.
    Uses Google's Gemini API for object detection in camera frames.
    """
    
    def __init__(
        self,
        camera_index=1, 
        api_key=os.getenv('GOOGLE_API_KEY'),
        model='gemini-2.0-flash-exp',
        COM_PORT=None,
        system_instructions=DEFAULT_SYSTEM_INSTRUCTIONS
    ):
        """
        Initialize the VideoUtilities class.
        
        Args:
            camera_index: Index of the camera to use
            api_key: Google API key for Gemini
            model: Gemini model to use
            COM_PORT: Serial port for Dobot connection
            system_instructions: Instructions for the vision model
        """
        if not api_key:
            raise ValueError('API key is required')
        
        if not system_instructions:
            raise ValueError('System instructions are required')
        
        self.model = model
        self.COM_PORT = COM_PORT
        self.camera_index = camera_index
        self.camera = None
        self.dobot = None
        self.system_instructions = system_instructions
        
        self._initialize_client(api_key)
    
    def _initialize_client(self, api_key):
        """Initialize the Gemini API client."""
        try:
            self.client = Client(
                api_key=api_key,
                http_options={'api_version': 'v1alpha'}
            )
            logger.info('Gemini client initialized successfully')
            
        except Exception as e:
            logger.error(f'Failed to initialize client: {e}')
            raise ValueError('Client initialization failed')
        
    def __enter__(self):
        """Set up resources when entering context manager."""
        self._setup_camera()
        
        if self.COM_PORT is not None:
            self._setup_dobot()
            
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager."""
        self._cleanup_camera()
        if self.dobot:
            self._cleanup_dobot()
        
    def _setup_camera(self):
        """Initialize the camera."""
        logger.info('Initializing camera')
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            raise ValueError(f'Failed to open camera at index {self.camera_index}')
        logger.info('Camera initialized successfully')
        
    def _setup_dobot(self):
        """Initialize the Dobot robot arm."""
        logger.info('Initializing Dobot')
        try:
            self.dobot = Dobot(
                port=self.COM_PORT,
                verbose=True
            )
            logger.info('Dobot initialized successfully')
            
        except Exception as e:
            logger.error(f'Failed to initialize Dobot: {e}')
            raise ValueError('Dobot initialization failed')
        
    def _cleanup_camera(self):
        """Release camera resources."""
        if self.camera and self.camera.isOpened():
            self.camera.release()
            logger.info('Camera released')
        
    def _cleanup_dobot(self):
        """Close connection to Dobot."""
        if self.dobot:    
            self.dobot.close()
            logger.info("Dobot connection closed")
        
    def _convert_image_to_base64(self, frame: np.ndarray) -> dict:
        """
        Convert a numpy image array to base64 encoded JPEG.
        
        Args:
            frame: OpenCV image frame (numpy array)
            
        Returns:
            dict: Image data in format required by Gemini API
        """
        with BytesIO() as buffer:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail([800, 800])
            image.save(buffer, format='JPEG')
            buffer.seek(0)
            
            return {
                'mime_type': 'image/jpeg', 
                'data': b64encode(buffer.read()).decode()
            }

    def _parse_json_response(self, response_text: str) -> dict:
        """
        Extract and parse JSON from model response.
        
        Args:
            response_text: Text response from Gemini model
            
        Returns:
            dict: Parsed JSON object or original text if parsing fails
        """
        try:
            # Extract JSON from markdown code blocks if present
            lines = response_text.splitlines()
            for i, line in enumerate(lines):
                if line == '```json':
                    json_text = '\n'.join(lines[i+1:])
                    json_text = json_text.split("```")[0]
                    break
            return json.loads(json_text)
            
        except Exception as e:
            logger.warning(f'Failed to parse JSON: {e}')
            return response_text
        
    def get_coords_from_camera(self):
        """
        Capture an image and use Gemini to detect object coordinates.
        
        Returns:
            dict: Detected object information with coordinates
        """
        success, frame = self.camera.read()
        if not success:
            logger.error("Failed to capture frame from camera")
            return None
        
        logger.info("Converting frame to base64")            
        image_data = self._convert_image_to_base64(frame)
        
        logger.info("Sending image to Gemini")
        response = self.client.models.generate_content(
            model = self.model,
            contents=[
                self.system_instructions,
                Part.from_bytes(**image_data)
            ],
            config = GenerateContentConfig(
                temperature = 0.3
            )
        )
        
        logger.info(f"Received Gemini response: {response.text}")
        return (response.text)

    def json_output_to_robot_coordinates(self, response: dict) -> dict:
        """
        Convert detected object coordinates to robot coordinates.
        
        Args:
            response: JSON response from Gemini with object coordinates
            
        Returns:
            dict: Object coordinates in robot frame of reference
        """
        (x1, y1, z1, _, _, _, _, _) = self.dobot.pose()
        
        x = response['position']['x']
        y = response['position']['y']  
        z = response['position']['z']  

        Robot = {
            'Robot': {
                'x': x + x1,
                'y': y + y1,
                'z': z + z1
            }
        }

        logger.info(f"Robot coordinates: {Robot}")
        return Robot