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
        
    