from livekit.agents.llm import FunctionContext, ai_callable, TypeInfo
from .utilities import DobotVision
from typing import Annotated
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class AssistantFunctions(FunctionContext):
    
    def __init__(self):
        
        logger.info('Initializing AssistantFunctions')

        super().__init__()
        self.video_utils = DobotVision(
            COM_PORT='COM5'
        )
        self.video_utils.__enter__()
        self.positions = self.video_utils.dobot.pose()
        
        logger.info('AssistantFunctions initialized successfully')
        
    @ai_callable(
        
        description = """

            Call this function once whenever the user asks anything about what u can 'see' or if they ask to 'open the camera and tell me ...'  or they say 'Tell me what i'm currently holding'.
            This function will return a string with json data describing the positions of objects detected by the camera. Use this data to describe what you see in a natural way to the user and don't use 
            any artifical words or responses returned by the AI.
            
            
        """
    )
    def get_objects(self):
        try:
            objects = self.video_utils.get_coords_from_camera()
            logger.info(f'Objects detected: {objects}')
        
            return f"Detected Objects: {objects}"
        
        except Exception as e:
            logger.error(f'Error in get_objects: {e}')
            return "Sorry, I can't detect any objects and i've encountered an error"
    
    @ai_callable(
        description = '''
        
        This function is used to move the robot to the position of an object detected by the camera.
        Call this function whenever the user asks to 'move the robot to some specific object detected by the camera'.
        
        for example, If the user says 'move the robot to the red cube' you should call this function with the argument 'red_cube' and so on
        
        '''
    )
    def move_robot_from_object_positions(
        self,
        object: Annotated[
            str,
            TypeInfo(description = 'the name of the object mentioned by the user')
        ]
    ):
        objects = self.video_utils.get_coords_from_camera()
        logger.info(f'Objects: {objects}, Object: {object}')
        
        if object in objects:
            
            robot_coords = self.video_utils.json_output_to_robot_coordinates(objects[object])
            x = robot_coords['Robot']['x']
            y = robot_coords['Robot']['y']
            z = robot_coords['Robot']['z']
            
            logger.info(f'Robot Coords: {robot_coords}')
            
            
            self.video_utils.dobot.move_to(
                x = x,
                y = y,
                z = z,
                r = 0,
                wait = True
            )
            return f'Moving robot to {x}, {y}, {z}'
        
        else:
            return f" I'm sorry but i can't find {object} from the camera"
        
    
    @ai_callable(
        description="""
        
        This function moves the robot to a specified joint angle or angles.
        Parameters:
        - base: The base angle of the robot must be in between -100 and 100
        - shoulder: The shoulder angle of the robot must be in between 0 to 75
        - elbow: The elbow angle of the robot must be in between 0 to 65
        - end_effector: The end-effector angle of the robot must be in between 0 to 100
        - end_effector_state: The end-effector state of the robot, this could be either True or False or  on or off
        
        Call this function whenever the user asks you to move any of the robot's joints to a particular angle.
        if the user only specifies the end-effector state, Then keep the other angles as None.
        
        if the user only specifies one angle, then keep the other angles as None
        
        if the user asks u to move the robot back to it's 'home' position then move the robot's base angle to 1, shoulder angle to 1, elbow angle to 1 and end-effector angle to 1 and set the end_effector state to False unless a different state is specified.
        
        """
    )
    def move_robot(
        self,
        
        base: Annotated[
            int, TypeInfo(description="The base angle of the robot"),
        ] = None,
        
        shoulder: Annotated[
            int, TypeInfo(description="The shoulder angle of the robot"),
        ] = None,
        
        elbow: Annotated[
            int, TypeInfo(description="The elbow angle of the robot"),
        ] = None,
        
        end_effector: Annotated[
            int, TypeInfo(description="The end-effector angle of the robot"),
        ] = None,
        
        end_effector_state: Annotated[
            bool, TypeInfo(description="The end-effector state of the robot, this could be either True or False"),
        ] = False
        
    ) -> str:
        """
        Called when the user asked to move the robot to a specific joint angle
        """
        logger.info(f"Moving robot to {base}, {shoulder}, {elbow}, {end_effector}")
        
        self.video_utils.dobot._set_ptp_cmd(
            base or self.positions[4],
            shoulder or self.positions[5],
            elbow or self.positions[6],
            end_effector or self.positions[7],
            mode = 4,
            wait = False
        )
        
        #Update the positions of the robot
        self.positions = self.video_utils.dobot.pose()
        logger.info(f"Robot moved to {self.positions}")
        
        if end_effector_state:
            self.video_utils.dobot.suck(True)
        else:
            self.video_utils.dobot.suck(False)
            
        return f"I've Moved the robot's Base: {base}, Shoulder: {shoulder}, elbow: {elbow}, end_effector: {end_effector}"
    
    @ai_callable(
        description = """
        
            This function is used to cleanup the program's resources. Call this function whenever the user
            says 'close the program or camera or robot' or something similar.
        
        """
    )
    def close(self):
        self.video_utils.__exit__(None, None, None)