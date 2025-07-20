import pdb
import random
from time import time, sleep
import qi

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

class Pepper:
    def __init__(self):
        self.session = qi.Session()
        self.is_connected = False
        
    def __del__(self):
        # Cleanup actions when the object is destroyed
        self.behavior_mng_service.stopAllBehaviors()        
        if self.is_connected:
            print("Disconnecting from the robot...")
            # self.video_proxy.unsubscribe(self.subscriber_id)
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
        self.autonomous_moves = self.session.service("ALAutonomousMoves")

    def execute_animation(self, action):
        """
        Execute animation with optional speech.
        """
        if not self.is_connected:
            return
        
        self.behavior_mng_service.stopAllBehaviors()
        if action:
            self.behavior_mng_service.startBehavior(action)


    # To update volume
    def update_volume(self, volume): 
        volume_n = round(max(0, volume/10), 1)
        # print(f"Volume_n: [volume, volume_n]")
        self.tts.setVolume(volume_n)
        
        if volume == 0 or volume == 1 or volume == 2 or volume == 3:
            messages_1= [
                "Humm",
                "Hummmmm",
                "Ha",
                "Uh-huh",
                "Uhh",
                "Uh-huh",
                "Huh",
                "Boop",
                "Meek meek",
                "Ohh",
                "Uhm",
                "Ehemmmm",
                "Ehh",
                "Ehem",
                "Uh",
                "Beep"               
            ]
            
            random_1 = random.choice(messages_1)
            self.tts.say(random_1)
            
        elif volume == 4 or volume == 5 or volume == 6:
            messages_2= [
                "Ohhhhh",
                "Ahhhh",
                "Cold room",
                "Uahhh",
                "Very peaceful",
                "All calm",
                "Nice atmosphere",
                "Looks compfy",
                "It's a cold room",
                "Weather's cold",
                "Feels chilly",
                "Just chillin'",
                "Feels nice",
                "All peaceful",
                "Seems calm",
                "Good energy",
                "Good vibes",
                "Pleasant place",
                "Nice vibe",
                "Looking good"                  
            ]
            
            random_2 = random.choice(messages_2)
            self.tts.say(random_2)
            
        elif volume == 7 or volume == 8 or volume == 9 or volume == 10:
            messages_3= [
                "I want your attention right now",
                "Look at me please",
                "This is really important",
                "I need you to focus on me",
                "Stop what you're doing",
                "Listen to me right now",
                "I have something urgent",
                "Please give me your attention",
                "I really need to talk to you",
                "Can you look at me immediately",
                "I need your full attention",
                "Hey, this can't wait any longer",
                "Drop everything and listen",
                "I'm talking to you right now",
                "You need to hear this",
                "Put that down and focus",
                "I'm trying to get your attention",
                "Hello, this is extremely urgent",
                "I need you to stop and listen",
                "Please turn around and look at me"                    
            ]
            
            random_3 = random.choice(messages_3)
            self.tts.say(random_3)
            
        
    # To update combined movements (gesture + head movement)
    def update_movement(self, gesture, head_mov):
        # Combine gesture and head movement: gesture * 10 + head_movement
        combined_movement = head_mov * 10 + gesture 
        print(f"Combined Movement (Gesture: {gesture}, Head: {head_mov}): {combined_movement}")
        self.behavior_mng_service.stopAllBehaviors()
        self.behavior_mng_service.startBehavior("action_gcrl/" + str(combined_movement))
    
    # To update navigation with proper mapping
    def update_navigation(self, navigation):
        # Map navigation 0-4 to 30-34
        navigation_action = 30 + navigation
        print(f"Navigation: {navigation} -> Action: {navigation_action}")
        self.autonomous_moves.setBackgroundStrategy("none")
        self.behavior_mng_service.stopAllBehaviors()
        self.behavior_mng_service.startBehavior("action_gcrl/" + str(navigation_action))
    
    # Function to execute an action with proper sequence
    def execute_action(self, navigation, gestures, head_mov, volume):

        self.update_navigation(navigation)
        sleep(1)  # Wait for navigation to complete
        
        # Then execute combined gesture and head movement
        self.update_movement(gestures, head_mov)
        
        # Update volume
        self.update_volume(volume)
        
    def update_behavior(self, delta_head, delta_nav, delta_gesture, delta_volume, head_m, nav, gesture, volume):
    # Convert integer deltas to strings to match the existing logic
        delta_nav = str(delta_nav)
        delta_gesture = str(delta_gesture)
        delta_head = str(delta_head)
        delta_volume = str(delta_volume)
        
        # Navigation updates
        if delta_nav == "1":
            nav = min(4, nav + 1)
        elif delta_nav == "-1":
            nav = max(0, nav - 1)
        # Keep same for "0" - nav stays unchanged
            
        # Gesture updates
        if delta_gesture == "1":
            gesture = min(9, gesture + 1)
        elif delta_gesture == "-1":
            gesture = max(0, gesture - 1)
        # Keep same for "0" - gesture stays unchanged
                
        # Head movement updates
        if delta_head == "1":
            head_m = min(2, head_m + 1)
        elif delta_head == "-1":
            head_m = max(0, head_m - 1)
        # Keep same for "0" - head_m stays unchanged
        
        # Volume updates
        if delta_volume == "1":
            volume = min(10, volume + 1)
        elif delta_volume == "-1":
            volume = max(0, volume - 1)
        # Keep same for "0" - volume stays unchanged
        
        print(f"Navigation: {nav}, Gesture: {gesture}, Head_Movement: {head_m}, Volume: {volume}")
        
        # Perform the action
        # self.execute_action(nav, gesture, head_m, volume)
        
        return head_m, nav, gesture, volume
        

        
