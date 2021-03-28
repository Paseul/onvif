import sys
sys.path.append('/home/jh/.virtualenvs/onvif/lib/python3.6/site-packages/wsdl')
from onvif import ONVIFCamera
from time import sleep

IP = "192.168.1.64"  # Camera IP address
PORT = 80  # Port
USER = "admin"  # Username
PASS = "laser123"  # Password

class ptzControl():
    def __init__(self):
    #Several cameras that have been tried  -------------------------------------
    #Netcat camera (on my local network) Port 8899
        self.mycam = ONVIFCamera(IP, PORT, USER, PASS) #, '/home/jh/.virtualenvs/onvif/lib/python3.6/site-packages/wsdl')
    #This is a demo camera that anyone can use for testing
    #Toshiba IKS-WP816R
        #self.mycam = ONVIFCamera('67.137.21.190', 80, 'toshiba', 'security', '/etc/onvif/wsdl/')

        # Create media service object
        self.media = self.mycam.create_media_service()
        # Get target profile
        self.media_profile = self.media.GetProfiles()[0]
        # Use the first profile and Profiles have at least one
        token = self.media_profile.token

    #PTZ controls  -------------------------------------------------------------
        self.ptz = self.mycam.create_ptz_service()

        #Get available PTZ services
        request = self.ptz.create_type('GetServiceCapabilities')
        Service_Capabilities = self.ptz.GetServiceCapabilities(request)

        #Get PTZ status
        status = self.ptz.GetStatus({'ProfileToken':token})

        # Get PTZ configuration options for getting option ranges
        request = self.ptz.create_type('GetConfigurationOptions')
        request.ConfigurationToken = self.media_profile.PTZConfiguration.token
        ptz_configuration_options = self.ptz.GetConfigurationOptions(request)

        self.requestc = self.ptz.create_type('ContinuousMove')
        self.requestc.ProfileToken = self.media_profile.token
        if self.requestc.Velocity is None:
            self.requestc.Velocity = self.ptz.GetStatus({'ProfileToken': self.media_profile.token}).Position
            self.requestc.Velocity.PanTilt.space = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].URI
            self.requestc.Velocity.Zoom.space = ptz_configuration_options.Spaces.ContinuousZoomVelocitySpace[0].URI

        self.requesta = self.ptz.create_type('AbsoluteMove')
        self.requesta.ProfileToken = self.media_profile.token

        if self.requesta.Position is None:
            self.requesta.Position = self.ptz.GetStatus({'ProfileToken': self.media_profile.token}).Position
        if self.requesta.Speed is None:
            self.requesta.Speed = self.ptz.GetStatus({'ProfileToken': self.media_profile.token}).Position

        self.requestr = self.ptz.create_type('RelativeMove')
        self.requestr.ProfileToken = self.media_profile.token
        if self.requestr.Translation is None:
            self.requestr.Translation = self.ptz.GetStatus({'ProfileToken': self.media_profile.token}).Position
            self.requestr.Translation.PanTilt.space = ptz_configuration_options.Spaces.RelativePanTiltTranslationSpace[0].URI
            self.requestr.Translation.Zoom.space = ptz_configuration_options.Spaces.RelativeZoomTranslationSpace[0].URI
        if self.requestr.Speed is None:
            self.requestr.Speed = self.ptz.GetStatus({'ProfileToken': self.media_profile.token}).Position

        self.requests = self.ptz.create_type('Stop')
        self.requests.ProfileToken = self.media_profile.token

        self.requestp = self.ptz.create_type('SetPreset')
        self.requestp.ProfileToken = self.media_profile.token

        self.requestg = self.ptz.create_type('GotoPreset')
        self.requestg.ProfileToken = self.media_profile.token

        self.stop()

#Stop pan, tilt and zoom
    def stop(self):
        self.requests.PanTilt = True
        self.requests.Zoom = True
        #print self.requests
        self.ptz.Stop(self.requests)

#Continuous move functions
    def perform_move(self, requestc):
        # Start continuous move
        ret = self.ptz.ContinuousMove(requestc)

    def move_tilt(self, velocity):
        self.requestc.Velocity.PanTilt.x = 0.0
        self.requestc.Velocity.PanTilt.y = velocity
        self.perform_move(self.requestc)

    def move_pan(self, velocity):
        self.requestc.Velocity.PanTilt.x = velocity
        self.requestc.Velocity.PanTilt.y = 0.0
        self.perform_move(self.requestc)

    def move_continuous(self, pan, tilt):
        self.requestc.Velocity.PanTilt.x = pan
        self.requestc.Velocity.PanTilt.y = tilt
        self.perform_move(self.requestc)

    def zoom(self, velocity):
        self.requestc.Velocity.Zoom.x = velocity
        self.perform_move(self.requestc)

#Absolute move functions --NO ERRORS BUT CAMERA DOES NOT MOVE
    def move_abspantilt(self, pan, tilt, velocity):
        self.requesta.Position.PanTilt.x = pan
        self.requesta.Position.PanTilt.y = tilt
        self.requesta.Speed.PanTilt.x = velocity
        self.requesta.Speed.PanTilt.y = velocity
        ret = self.ptz.AbsoluteMove(self.requesta)

#Relative move functions --NO ERRORS BUT CAMERA DOES NOT MOVE
    def move_relative(self, pan, tilt, velocity):
        self.requestr.Translation.PanTilt.x = pan
        self.requestr.Translation.PanTilt.y = tilt
        self.requestr.Speed.PanTilt.x = velocity
        self.requestr.Speed.PanTilt.y = velocity
        ret = self.ptz.RelativeMove(self.requestr)

    def zoom_relative(self, zoom, velocity):
        self.requestr.Translation.PanTilt.x = 0
        self.requestr.Translation.PanTilt.y = 0
        self.requestr.Translation.Zoom.x = zoom
        self.requestr.Speed.PanTilt.x = 0
        self.requestr.Speed.PanTilt.y = 0
        self.requestr.Speed.Zoom.x = velocity
        ret = self.ptz.RelativeMove(self.requestr)

#Sets preset set, query and and go to
    def set_preset(self, name):
        self.requestp.PresetName = name
        self.requestp.PresetToken = '1'
        self.preset = self.ptz.SetPreset(self.requestp)  #returns the PresetToken

    def get_preset(self):
        self.ptzPresetsList = self.ptz.GetPresets(self.requestc)

    def goto_preset(self, name):
        self.requestg.PresetToken = '1'
        self.ptz.GotoPreset(self.requestg)