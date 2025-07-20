import math
import random
import time
import qi
# from gcrl_formulation import *

# Camera resolution constants
kQQVGA = 160  # 160x120
kQVGA = 320  # 320x240
kVGA = 640  # 640x480
k4VGA = 1280  # 1280x960
k16VGA = 2560  # 2560x1920

# Color spaces constants
kYUV422ColorSpace = 9
kRGBColorSpace = 11
kBGRColorSpace = 13
kHSVColorSpace = 15

# config = multi_goal_config

class Pepper:
    def __init__(self):
        self.session = qi.Session()
        self.is_connected = False
        
    def __del__(self):
        # Cleanup actions when the object is destroyed
        self.behavior_mng_service.stopAllBehaviors()        
        if self.is_connected:
            print("Disconnecting from the robot...")
            self.video_proxy.unsubscribe(self.subscriber_id)
            self.session.close()
            print("Session closed.")
    
    def connect(self, ip, port):
        # Connect to the robot
        print("Connect to the robot...")
        try:
            self.session.connect("tcp://{0}:{1}".format(ip, port))
            print("Session Connected....!")
            self.is_connected = True
        except Exception as e:
            print("Could not connect to Pepper:", e)
            self.is_connected = False
            return
        
        self.tts = self.session.service("ALTextToSpeech")
        self.leds = self.session.service("ALLeds")

        
        self.video_proxy = self.session.service("ALVideoDevice") #, PEPPER_IP, PORT)
        
        self.camera_id = 0  # 0 = Top Camera, 1 = Bottom Camera
        self.resolution = k4VGA #--switched#kQVGA  # 320x240 resolution
        self.color_space = kBGRColorSpace  # OpenCV expects BGR format
        self.fps = 5  # Frames per second
        
        # Subscribe to Pepper’s camera
        self.subscriber_id = self.video_proxy.subscribeCamera("pepper_cam", 
                                                              self.camera_id, 
                                                              self.resolution, 
                                                              self.color_space, 
                                                              self.fps)

        self.posture_service = self.session.service("ALRobotPosture")
        self.posture_service.goToPosture("StandInit", 1.0)
        
        self.behavior_mng_service = self.session.service("ALBehaviorManager")

        self.motion_service = self.session.service("ALMotion")

    # To update volume
    def update_volume(self, volume, state):    
        volume_n = round(max(0, volume/10), 1)
        # print(f"Volume_n: [volume, volume_n]")
        self.tts.setVolume(volume_n)
        
        if state != 0 and volume == 10:
            messages = [
                "Yes, stare into my starry eyes",
                "Look at me, that's right",
                "Yes, I am more important",
                "Oh so you are looking",
                "So you can see me",
                "Just wanted your attention",                
            ]    
            random_greeting = random.choice(messages)
            self.tts.say(random_greeting) 
        
        
        if volume == 1:
            messages_1= [
                "Hey",
                "Hi",
                "Hello",
                "Heyyy"                    
            ]
            
            random_1 = random.choice(messages_1)
            self.tts.say(random_1)
            
        elif volume == 2:
            messages_2= [
                "Hello there",
                "Hi there",
                "Hey there"                    
            ]
            
            random_2 = random.choice(messages_2)
            self.tts.say(random_2)
            
        elif volume == 3:
            messages_3= [
                "Hello, Human!",
                "Hi, Human!",
                "Hey, Human!"                    
            ]
            
            random_3 = random.choice(messages_3)
            self.tts.say(random_3)
        
        elif volume == 4:
            messages_4= [
                "Can you hear me?",
                "Are you listening?",
                "Can you see me?",
                "Do you notice me?",
                "Do you see me?",
                "Can you notice me?"                    
            ]
            
            random_4 = random.choice(messages_4)
            self.tts.say(random_4) 
            
        elif volume == 5:
            messages_5= [
                "Are you busy?",
                "Am I interrupting?",
                "You look busy",
                "Are you okay?",
                "Excuse me",
                "Busy with the video?"                  
            ]
            
            random_5 = random.choice(messages_5)
            self.tts.say(random_5)   
            
        elif volume == 6:
            messages_6= [
                "Can you please look at me",
                "Would you mind looking my way?",
                "Can you give me a glance?",
                "Could you face me for a moment?",
                "Look over here for a sec?",
                "Mind making eye contact for a moment?"                 
            ]
            
            random_6 = random.choice(messages_6)
            self.tts.say(random_6)
        
        elif volume == 7:
            messages_7= [
                "Hello, Stop ignoring me!",
                "Wow, just gonna ignore me like that?",
                "I’m not invisible, you know!",
                "Just gonna leave me hanging huh?",
                "Hello?? A response would be nice!",
                "Why are you ghosting me"                  
            ]
            
            random_7 = random.choice(messages_7)
            self.tts.say(random_7)
        
        elif volume == 8:
            messages_8= [
                "Talk to me about your video human!",
                "What’s the deal about your video?",
                "What’s so special about your video?",
                "I’m just dying to hear about your video!",
                "Wow me, human! What’s your video all about?"                
            ]
            
            random_8 = random.choice(messages_8)
            self.tts.say(random_8)
        
        elif volume == 9:
            messages_9= [
                "Attention, humannn!!",
                "Earthling! I want your attention!",
                "Acknowledge my existence, human!",
                "I’m just dying to hear about your video!",
                " I require your focus!"                
            ]
            
            random_9 = random.choice(messages_9)
            self.tts.say(random_9)
            
        elif state == 0 and volume == 10:
            messages_10= [
                "I said give me attention right now!",
                "Why are you not looking at me!",
                "I am more important than your video",
                "Seriously, look at me!",
                "Keep giving me attention",
                "Come on, look at me",
                "I am dying for your attention here",
                "I need your attention now!",
                "Stop what you're doing and listen!",
                "Focus on me right now!" ,
                "A video huh? More important?"               
            ]
            
            random_10 = random.choice(messages_10)
            self.tts.say(random_10)
        
    # To update movements
    def update_movements(self, movement):
        print(f"Movement: {movement}")
        # self.behavior_mng_service.stopAllBehaviors()
        # self.behavior_mng_service.startBehavior("action_gcrl/" + str(movement)) 

    # To update head_movements
    def update_head_movements(self, position):
        print
        """
        Moves Pepper's head to a predefined orientation.
        - 0: Looking down
        - 1: Looking halfway/front
        - 2: Looking fully forward
        """

        effectorName = "Head"
        self.motion_service.wbEnableEffectorControl(effectorName, True)

        # Define target angles in degrees
        # Format: [x, y, z] in degrees
        targets_deg = {
            0: [0.0, -40.0, 0.0],   # Looking down
            1: [0.0, -15.0, 0.0],   # Looking halfway/front
            2: [0.0, 0.0, 0.0],     # Looking front
        }

        # Ensure valid input
        if position not in targets_deg:
            print("Invalid position. Use 0 (down), 1 (halfway), or 2 (front).")
            return

        # Convert degrees to radians
        target_rad = [angle * math.pi / 180.0 for angle in targets_deg[position]]
        
        # Move head
        self.motion_service.wbSetEffectorControl(effectorName, target_rad)
        time.sleep(3.0)

        # Disable control and go to rest
        self.motion_service.wbEnableEffectorControl(effectorName, False)
        self.motion_service.rest()

    # Function to execute an action
    def execute_action(self, light, movement, volume):
        self.update_lights(light)
        self.update_movements(movement)
        self.update_volume(volume)
        
    def update_behavior(self, action, light, movement, volume):
        # pdb.set_trace()
        l_action, m_action, v_action = action.split(', ')
        
        if l_action == "Increase L":
            light = min(10, light + 1)
        elif l_action == "Decrease L":
            light = max(0, light - 1)
            
        if m_action == "Increase M":
            movement = min(10, movement + 1)
        elif m_action == "Decrease M":
            movement = max(0, movement - 1)
                
        if v_action == "Increase V":
            volume = min(10, volume + 1)
        elif v_action == "Decrease V":
            volume = max(0, volume - 1)
                
        # Keep Same
        if l_action == "Keep L":
            light = light
        elif m_action == "Keep M":
            movement = movement
        elif v_action == "Keep V":
            volume = volume
        
        print(f"Light: {light}, Movement: {movement}, Volume: {volume}")
        
        # Perform the action
        self.execute_action(light, movement, volume)
        return light, movement, volume     
        

        
