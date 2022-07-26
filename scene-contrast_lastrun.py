#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on July 26, 2022, at 14:25
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, iohub, hardware
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'Forced_Response_scene_contrast'  # from the Builder filename that created this script
expInfo = {'participant': '', 'gender': ['male', 'female', 'other', 'do not want to say'], 'age': ''}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\hanzh\\Desktop\\Pupillometry Demo\\scene-contrast_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1600, 1200], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='Lab', color='black', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='pix')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "ins1"
ins1Clock = core.Clock()
import json
import imageio
from math import cos, exp, pow, sqrt, pi
import numpy as np

def createBullseye(target_loc,factor,img_array):
    #from math import cos, exp, pow, sqrt, pi
    #import numpy as np
    #target_location(x,y) tuple - location of bullseye target
    #factor [0+] - how much you want the bullseye function

    img_array = img_array/255.0 #normalize to 0-1
    
    s_p = 20
    X = np.tile( np.linspace( 1, s_p, s_p), (s_p, 1) )
    Y = X.T
    c = (s_p + 1.0) / 2 if s_p%2 == 0 else s_p/2
    cdist_grid = np.sqrt((X-c)**2 + (Y-c)**2)

    #gaussian:
    sigma = 5
    x = cdist_grid
    gauss_grid = np.exp( -(x**2) / (2*sigma**2) )
    # sine
    l = 3
    x = (cdist_grid / np.max(cdist_grid)) * l*2*np.pi
    sin_grid = np.sin(x)

    # combine 
    bullseye = gauss_grid * sin_grid
    bullseye = bullseye / np.max(bullseye) # make sure peak == 1

    # determine which part of the bullseye should brighten, which should darken:
    brighten = np.array(bullseye > 0, dtype=int)
    darken = np.array(bullseye < 0, dtype=int)

    #patch = img_array[target_loc[1]:target_loc[1]+s_p,target_loc[0]:target_loc[0]+s_p]
    patch = img_array[target_loc[1]-int(s_p/2):target_loc[1]+int(s_p/2),target_loc[0]-int(s_p/2):target_loc[0]+int(s_p/2)]

    factor = factor    #Wouter's original was 2.5

    # brighten patch:
    # bullseye * 1/2distance from one:
    sd = np.std(patch)
    m = np.mean(patch)

    if sd < 0.06:
        print("too low sd:")
        sd = 0.08
    elif sd > .35:
        print("too high sd:")
        sd = 0.17 #use the median stdev value of all patches across all images
    c = sd/m
    #print c, m, sd
    brightened = brighten * (patch + bullseye * (1 - patch) * factor * sd)
    # darken patch:
    # bullseye * 1/2distance from zero:
    darkened = darken * (patch + bullseye * (patch) * factor * sd)
    # set in the image
    img_array[target_loc[1]:target_loc[1]+s_p,target_loc[0]:target_loc[0]+s_p] = brightened + darkened
    #mimg = Image.fromarray(img_array * 256)
    #mimg = Image.convert("F",mimg)
    mimg = img_array * 255
    mimg = np.asarray(mimg)
    mimg = mimg.astype(np.uint8)
    return mimg, sd
    

def check_location(sac, target_loc, r):
    start_x = sac.start_gaze_x
    start_y = sac.start_gaze_y
    end_x = sac.end_gaze_x
    end_y = sac.end_gaze_y
    dist = math.hypot(end_x - target_loc[0], end_y - target_loc[1])
    if dist <= r:
        found = 1
    else:
        found = 0
    return found
    
def check_timing(sac, origin):
    saccade_duration = sac.duration
    saccade_endtime = sac.time
    SRT = saccade_endtime - saccade_duration - origin
    #PT = 1000*(SRT - target_onset)
    if (abs(SRT - 3) < .2):
        timing = 'ontime'
    elif SRT - 3 < -.2:
        timing = 'fast'
    else:
        timing = 'slow'
    return timing
# Setup eyetracking
import psychopy.iohub as io
if expInfo['participant']=='':
    fname = 'EXPFILE'
else:
    fname = expInfo['participant']
ioDevice = 'eyetracker.hw.sr_research.eyelink.EyeTracker'
ioConfig = {
    ioDevice: {
        'name': 'tracker',
        'model_name': 'EYELINK 1000 REMOTE',
        'simulation_mode': False,
        'network_settings': '100.1.1.1',
        'default_native_data_file_name': fname,
        'runtime_settings': {
            'sampling_rate': 500.0,
            'track_eyes': 'RIGHT_EYE',
            'sample_filtering': {
                'sample_filtering': 'FILTER_LEVEL_2',
                'elLiveFiltering': 'FILTER_LEVEL_1',
            },
            'vog_settings': {
                'pupil_measure_types': 'PUPIL_DIAMETER',
                'tracking_mode': 'PUPIL_CR_TRACKING',
                'pupil_center_algorithm': 'ELLIPSE_FIT',
            }
        },
        'calibration': {
            'type': 'FIVE_POINTS', 
            'color_type': None, 
            'unit_type': None, 
            'auto_pace': False, 
            'target_duration': 1.5, 
            'target_delay': 0.75, 
            'pacing_speed': 1.0, 
            'screen_background_color': [-1, -1, -1], 
            'target_type': 'CIRCLE_TARGET', 
            'target_attributes': {'outer_diameter': 40.0, 'inner_diameter': 20.0, 'outer_stroke_width': 2.0, 'outer_fill_color': [-1, -1, -1], 'outer_line_color': [-1., -1., -1.], 'inner_stroke_width': 2.0, 'inner_fill_color': [1, 1, 1], 'inner_line_color': [1., 1., 1.], 'outer_color': None, 'inner_color': None}
            }
    }
}
ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, experiment_code='Forced Response ColSing_Eye', session_code=ioSession, datastore_name=filename, **ioConfig)
eyetracker = ioServer.getDevice('tracker')

from psychopy.iohub.constants import EventConstants
import math
import psychtoolbox as ptb
from psychopy import sound

win.mouseVisible = False
#### Condition File ####
#cond_files = []
#shuffle(cond_files)
#condition_file = cond_files[0]

#### Numbers ####
# n of loops within each block
nReps_free = 1
nReps_timing = 1
nReps_forced = 1
# n of blocks
nBlocks = 4

#### Graphics ####
# size of fixation
size_fixation = 80
# instruction text size
text_size = 45
# text_wrap
text_wrap = 1280
ins1_text = visual.TextStim(win=win, name='ins1_text',
    text="You are about to begin a block of practice trials in a visual search game.\n\nOn each trial, a fixation cross will show at the screen center for you to stare at. Shortly after, an image will show on the screen. The image contains a small bull's eye target. Your task is to find the target. Once you have found it, press the spacebar. \n\nPlease let the experimenter know if you have any questions.\n\nPress the SPACE BAR to begin",
    font='Arial',
    pos=(0, 0), height=text_size, wrapWidth=text_wrap, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-3.0);
ins1_key = keyboard.Keyboard()

# Initialize components for Routine "recalibration"
recalibrationClock = core.Clock()

# Initialize components for Routine "start_recording"
start_recordingClock = core.Clock()
StartRecISI = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='StartRecISI')

# Initialize components for Routine "fixation_check"
fixation_checkClock = core.Clock()
fixation_cross = visual.ShapeStim(
    win=win, name='fixation_cross', vertices='cross',
    size=size_fixation,
    ori=0.0, pos=(0, 0),
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "trial_free"
trial_freeClock = core.Clock()
prev_target_found = -1
prev_emcount = -1
c = 3
image = visual.ImageStim(
    win=win,
    name='image', units='pix', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
key_resp = keyboard.Keyboard()

# Initialize components for Routine "feedback"
feedbackClock = core.Clock()
feedback_text = visual.TextStim(win=win, name='feedback_text',
    text='',
    font='Arial',
    pos=(0, 0), height=text_size, wrapWidth=text_wrap, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "warn_start_loc"
warn_start_locClock = core.Clock()
warn_text = visual.TextStim(win=win, name='warn_text',
    text='Stare at the screen center until the image shows up!',
    font='Arial',
    pos=(0, 0), height=text_size, wrapWidth=text_wrap, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "ITI"
ITIClock = core.Clock()
fixation_cross_2 = visual.ShapeStim(
    win=win, name='fixation_cross_2', vertices='cross',
    size=size_fixation,
    ori=0.0, pos=(0, 0),
    lineWidth=2.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)

# Initialize components for Routine "stop_recording"
stop_recordingClock = core.Clock()
StopRecISI = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='StopRecISI')

# Initialize components for Routine "end_study"
end_studyClock = core.Clock()
study_over = visual.TextStim(win=win, name='study_over',
    text='End of the task!',
    font='Arial',
    units='pix', pos=(0, 100), height=text_size, wrapWidth=text_wrap, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "ins1"-------
continueRoutine = True
# update component parameters for each repeat
ins1_key.keys = []
ins1_key.rt = []
_ins1_key_allKeys = []
# keep track of which components have finished
ins1Components = [ins1_text, ins1_key]
for thisComponent in ins1Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
ins1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "ins1"-------
while continueRoutine:
    # get current time
    t = ins1Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=ins1Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *ins1_text* updates
    if ins1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        ins1_text.frameNStart = frameN  # exact frame index
        ins1_text.tStart = t  # local t and not account for scr refresh
        ins1_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(ins1_text, 'tStartRefresh')  # time at next scr refresh
        ins1_text.setAutoDraw(True)
    
    # *ins1_key* updates
    waitOnFlip = False
    if ins1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        ins1_key.frameNStart = frameN  # exact frame index
        ins1_key.tStart = t  # local t and not account for scr refresh
        ins1_key.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(ins1_key, 'tStartRefresh')  # time at next scr refresh
        ins1_key.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(ins1_key.clock.reset)  # t=0 on next screen flip
    if ins1_key.status == STARTED and not waitOnFlip:
        theseKeys = ins1_key.getKeys(keyList=['space', '6'], waitRelease=False)
        _ins1_key_allKeys.extend(theseKeys)
        if len(_ins1_key_allKeys):
            ins1_key.keys = _ins1_key_allKeys[-1].name  # just the last key pressed
            ins1_key.rt = _ins1_key_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in ins1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "ins1"-------
for thisComponent in ins1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
if ins1_key.keys == '6':
    nReps_free = 0
# the Routine "ins1" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "recalibration"-------
continueRoutine = True
# update component parameters for each repeat
# recalibration
win.winHandle.minimize()
eyetracker.runSetupProcedure()
win.winHandle.activate()
win.winHandle.maximize()
# keep track of which components have finished
recalibrationComponents = []
for thisComponent in recalibrationComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
recalibrationClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "recalibration"-------
while continueRoutine:
    # get current time
    t = recalibrationClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=recalibrationClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in recalibrationComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "recalibration"-------
for thisComponent in recalibrationComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
win.winHandle.set_mouse_position(0,0)
win.flip()
# the Routine "recalibration" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "start_recording"-------
continueRoutine = True
routineTimer.add(0.500000)
# update component parameters for each repeat
#win.mouseVisible = False
eyetracker.setRecordingState(True)

# keep track of which components have finished
start_recordingComponents = [StartRecISI]
for thisComponent in start_recordingComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
start_recordingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "start_recording"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = start_recordingClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=start_recordingClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # *StartRecISI* period
    if StartRecISI.status == NOT_STARTED and t >= 0-frameTolerance:
        # keep track of start time/frame for later
        StartRecISI.frameNStart = frameN  # exact frame index
        StartRecISI.tStart = t  # local t and not account for scr refresh
        StartRecISI.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(StartRecISI, 'tStartRefresh')  # time at next scr refresh
        StartRecISI.start(.5)
    elif StartRecISI.status == STARTED:  # one frame should pass before updating params and completing
        StartRecISI.complete()  # finish the static period
        StartRecISI.tStop = StartRecISI.tStart + .5  # record stop time
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in start_recordingComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "start_recording"-------
for thisComponent in start_recordingComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('StartRecISI.started', StartRecISI.tStart)
thisExp.addData('StartRecISI.stopped', StartRecISI.tStop)

# set up handler to look after randomisation of conditions etc
free_loop = data.TrialHandler(nReps=nReps_free, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('prc_conditions.csv', selection='0:5'),
    seed=None, name='free_loop')
thisExp.addLoop(free_loop)  # add the loop to the experiment
thisFree_loop = free_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisFree_loop.rgb)
if thisFree_loop != None:
    for paramName in thisFree_loop:
        exec('{} = thisFree_loop[paramName]'.format(paramName))

for thisFree_loop in free_loop:
    currentLoop = free_loop
    # abbreviate parameter names if possible (e.g. rgb = thisFree_loop.rgb)
    if thisFree_loop != None:
        for paramName in thisFree_loop:
            exec('{} = thisFree_loop[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "fixation_check"-------
    continueRoutine = True
    # update component parameters for each repeat
    # subject must look at the fixation for 300 ms
    timer = core.CountdownTimer(1)
    
    if free_loop.finished == 0:
        blockN = 0
        trial_type = 'free'
        trialN = free_loop.thisN
    elif repeat_timing.finished == 0:
        blockN = repeat_timing.thisN
        trial_type = 'timing'
        trialN = timing_loop.thisN
    else:
        blockN = block_loop.thisN
        trial_type = 'forced'
        trialN = forced_loop.thisN
    
    # send event start signal
    eventmsg = '%s fixation start Block_%i Trial_%i' % (trial_type, blockN, trialN)
    eyetracker.sendMessage(eventmsg)
    ioServer.sendMessageEvent(text = eventmsg)
    print(eventmsg)
    # keep track of which components have finished
    fixation_checkComponents = [fixation_cross]
    for thisComponent in fixation_checkComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    fixation_checkClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "fixation_check"-------
    while continueRoutine:
        # get current time
        t = fixation_checkClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=fixation_checkClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_cross* updates
        if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_cross.frameNStart = frameN  # exact frame index
            fixation_cross.tStart = t  # local t and not account for scr refresh
            fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
            fixation_cross.setAutoDraw(True)
        # Get last eye position
        gpos = eyetracker.getLastGazePosition()
        
        if isinstance(gpos, (tuple, list)):
            # There was an eye position available
            # Check if it is within a visual stim called 'gaze_region_stim'
            #if fixation.contains(gpos):
            if math.hypot(gpos[0] - 0, gpos[1] - 0) <= size_fixation:
                pass
            else:
                timer.reset()
        else:
            timer.reset()
            
        if timer.getTime() <= 0:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in fixation_checkComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "fixation_check"-------
    for thisComponent in fixation_checkComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    free_loop.addData('fixation_cross.started', fixation_cross.tStartRefresh)
    free_loop.addData('fixation_cross.stopped', fixation_cross.tStopRefresh)
    # send event start signal
    eventmsg = '%s fixation end Block_%i Trial_%i' % (trial_type, blockN, trialN)
    eyetracker.sendMessage(eventmsg)
    ioServer.sendMessageEvent(text = eventmsg)
    print(eventmsg)
    # the Routine "fixation_check" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "trial_free"-------
    continueRoutine = True
    routineTimer.add(5.000000)
    # update component parameters for each repeat
    # read image
    path = 'practice_imgs/'+fname
    img = imageio.imread(path)
    
    #determine the multiplication factor based on number of eye movements
    #made on the previous trial.
    if prev_target_found == 1 and prev_emcount <= 3 and prev_emcount != -1:
        if c > 2:
            c = c - .2
        else:
            c = c #don't go below a factor of 2
    elif prev_target_found == 0 or prev_emcount > 6:
        c = c + .2
    else:
        c = c
    
    bimg, sd = createBullseye((target_x, target_y), c, img)
    imageio.imwrite('test.jpg', bimg)
    
    #image_stim = visual.ImageStim(win, image='test.jpg')
    #image_stim.draw()
    #win.flip()
    image.setImage('test.jpg')
    # clear events
    eyetracker.clearEvents()
    
    # used to decide whether to ignore the saccade right after the blink
    justBlinked = 0
    valid_first_saccade = False
    saccade_list = []
    
    # send event start signal
    eventmsg = '%s search start Block_%i Trial_%i' % (trial_type, blockN, trialN)
    eyetracker.sendMessage(eventmsg)
    ioServer.sendMessageEvent(text = eventmsg)
    print(eventmsg)
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    trial_freeComponents = [image, key_resp]
    for thisComponent in trial_freeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    trial_freeClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "trial_free"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = trial_freeClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=trial_freeClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image* updates
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            image.setAutoDraw(True)
        if image.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                image.tStop = t  # not accounting for scr refresh
                image.frameNStop = frameN  # exact frame index
                win.timeOnFlip(image, 'tStopRefresh')  # time at next scr refresh
                image.setAutoDraw(False)
        # get eye tracker events
        events = eyetracker.getEvents()
        
        for e in events:
            #ignore blinks
            if e.type == EventConstants.BLINK_END:
                justBlinked = 1
            # check for saccades
            elif e.type == EventConstants.SACCADE_END:
                # ignore the saccade ending event right after the blink
                if justBlinked == 1:
                    justBlinked = 0
                # save saccade
                else:   
                    saccade_list.append(e)
                    # if this is the first saccade
                    if len(saccade_list) == 1:
                        # check if saccade starts from the circle (could be more lenient)
                        if math.hypot(e.start_gaze_x - 0, e.start_gaze_y - 0) <= size_fixation:
                                valid_first_saccade = True
        
        # *key_resp* updates
        waitOnFlip = False
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.frameNStop = frameN  # exact frame index
                win.timeOnFlip(key_resp, 'tStopRefresh')  # time at next scr refresh
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['y', 'n', 'left', 'right', 'space'], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trial_freeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial_free"-------
    for thisComponent in trial_freeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    free_loop.addData('image.started', image.tStartRefresh)
    free_loop.addData('image.stopped', image.tStopRefresh)
    ## send event end signal
    eventmsg = '%s search end Block_%i Trial_%i' % (trial_type, blockN, trialN)
    eyetracker.sendMessage(eventmsg)
    ioServer.sendMessageEvent(text = eventmsg)
    print(eventmsg)
    
    # subjective report
    if len(key_resp.keys) != 0:
        report_found = True
    else:
        report_found = False
    
    # objective saccades
    saccade_found = []
    if len(saccade_list) != 0:
        for s in saccade_list:
            found = check_location(s, (target_x_py, target_y_py), 100)
            saccade_found.append(found)
        if (1 in saccade_found) & (report_found == True):
            prev_target_found = 1
            prev_emcount = saccade_found.index(1) + 1
            feedback_msg = 'Target Found!'
        else:
            prev_target_found = 0
            prev_emcount = len(saccade_list)
            feedback_msg = 'Target Not Found!'
    else:
        feedback_msg = 'No valid eye movements!'
        prev_target_found = -1
        prev_emcount = -1
    
            
    # check where did the subject look on first saccade
    if len(saccade_list) != 0:
        first_saccade = saccade_list[0]
        first_saccade_found = saccade_found[0]
        first_saccade_endtime = first_saccade.time
        first_saccade_duration = first_saccade.duration
        first_saccade_status = first_saccade.status
        SRT = first_saccade_endtime - first_saccade_duration - image.tStartRefresh # end time - duration -  stimulus onset
        
        start_x = first_saccade.start_gaze_x
        start_y = first_saccade.start_gaze_y
        end_x = first_saccade.end_gaze_x
        end_y = first_saccade.end_gaze_y
        if end_x <= 0:
            first_saccade_dest = 'left'
        else:
            first_saccade_dest = 'right'
    else:
        first_saccade_found = None
        first_saccade_endtime = None
        first_saccade_duration = None
        first_saccade_status = None
        SRT = None
        first_saccade_dest = None
        start_x = None
        start_y = None
        end_x = None
        end_y = None
       
    # log data
    thisExp.addData('valid_first_saccade', valid_first_saccade)
    thisExp.addData('first_saccade_duration', first_saccade_duration)
    thisExp.addData('first_saccade_endtime', first_saccade_endtime)
    thisExp.addData('first_saccade_status', first_saccade_status)
    thisExp.addData('SRT', SRT)
    thisExp.addData('first_saccade_found', first_saccade_found)
    thisExp.addData('first_saccade_dest', first_saccade_dest)
    thisExp.addData('start_x', start_x)
    thisExp.addData('start_y', start_y)
    thisExp.addData('end_x', end_x)
    thisExp.addData('end_y', end_y)
    thisExp.addData('feedback', feedback_msg)
    thisExp.addData('report_found', report_found)
    thisExp.addData('saccade_list', json.dumps(saccade_list))
    thisExp.addData('saccade_found', json.dumps(saccade_found))
    thisExp.addData('c_value', c)
    thisExp.addData('prev_target_found', prev_target_found)
    thisExp.addData('prev_emcount', prev_emcount)
    thisExp.addData('sd_value', sd)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    free_loop.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        free_loop.addData('key_resp.rt', key_resp.rt)
    free_loop.addData('key_resp.started', key_resp.tStartRefresh)
    free_loop.addData('key_resp.stopped', key_resp.tStopRefresh)
    
    # ------Prepare to start Routine "feedback"-------
    continueRoutine = True
    routineTimer.add(1.250000)
    # update component parameters for each repeat
    feedback_text.setText(feedback_msg)
    # keep track of which components have finished
    feedbackComponents = [feedback_text]
    for thisComponent in feedbackComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = feedbackClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=feedbackClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *feedback_text* updates
        if feedback_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            feedback_text.frameNStart = frameN  # exact frame index
            feedback_text.tStart = t  # local t and not account for scr refresh
            feedback_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_text, 'tStartRefresh')  # time at next scr refresh
            feedback_text.setAutoDraw(True)
        if feedback_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > feedback_text.tStartRefresh + 1.25-frameTolerance:
                # keep track of stop time/frame for later
                feedback_text.tStop = t  # not accounting for scr refresh
                feedback_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(feedback_text, 'tStopRefresh')  # time at next scr refresh
                feedback_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "feedback"-------
    for thisComponent in feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    # ------Prepare to start Routine "warn_start_loc"-------
    continueRoutine = True
    routineTimer.add(1.250000)
    # update component parameters for each repeat
    if valid_first_saccade:
        continueRoutine=False
    # keep track of which components have finished
    warn_start_locComponents = [warn_text]
    for thisComponent in warn_start_locComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    warn_start_locClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "warn_start_loc"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = warn_start_locClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=warn_start_locClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *warn_text* updates
        if warn_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            warn_text.frameNStart = frameN  # exact frame index
            warn_text.tStart = t  # local t and not account for scr refresh
            warn_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(warn_text, 'tStartRefresh')  # time at next scr refresh
            warn_text.setAutoDraw(True)
        if warn_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > warn_text.tStartRefresh + 1.25-frameTolerance:
                # keep track of stop time/frame for later
                warn_text.tStop = t  # not accounting for scr refresh
                warn_text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(warn_text, 'tStopRefresh')  # time at next scr refresh
                warn_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in warn_start_locComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "warn_start_loc"-------
    for thisComponent in warn_start_locComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    # ------Prepare to start Routine "ITI"-------
    continueRoutine = True
    routineTimer.add(2.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    ITIComponents = [fixation_cross_2]
    for thisComponent in ITIComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    ITIClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "ITI"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = ITIClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=ITIClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation_cross_2* updates
        if fixation_cross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            fixation_cross_2.frameNStart = frameN  # exact frame index
            fixation_cross_2.tStart = t  # local t and not account for scr refresh
            fixation_cross_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_cross_2, 'tStartRefresh')  # time at next scr refresh
            fixation_cross_2.setAutoDraw(True)
        if fixation_cross_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_cross_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                fixation_cross_2.tStop = t  # not accounting for scr refresh
                fixation_cross_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(fixation_cross_2, 'tStopRefresh')  # time at next scr refresh
                fixation_cross_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ITIComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "ITI"-------
    for thisComponent in ITIComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    free_loop.addData('fixation_cross_2.started', fixation_cross_2.tStartRefresh)
    free_loop.addData('fixation_cross_2.stopped', fixation_cross_2.tStopRefresh)
    thisExp.nextEntry()
    
# completed nReps_free repeats of 'free_loop'


# ------Prepare to start Routine "stop_recording"-------
continueRoutine = True
routineTimer.add(0.500000)
# update component parameters for each repeat
eyetracker.setRecordingState(False)
# keep track of which components have finished
stop_recordingComponents = [StopRecISI]
for thisComponent in stop_recordingComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
stop_recordingClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "stop_recording"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = stop_recordingClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=stop_recordingClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # *StopRecISI* period
    if StopRecISI.status == NOT_STARTED and t >= 0-frameTolerance:
        # keep track of start time/frame for later
        StopRecISI.frameNStart = frameN  # exact frame index
        StopRecISI.tStart = t  # local t and not account for scr refresh
        StopRecISI.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(StopRecISI, 'tStartRefresh')  # time at next scr refresh
        StopRecISI.start(.5)
    elif StopRecISI.status == STARTED:  # one frame should pass before updating params and completing
        StopRecISI.complete()  # finish the static period
        StopRecISI.tStop = StopRecISI.tStart + .5  # record stop time
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in stop_recordingComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "stop_recording"-------
for thisComponent in stop_recordingComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('StopRecISI.started', StopRecISI.tStart)
thisExp.addData('StopRecISI.stopped', StopRecISI.tStop)

# ------Prepare to start Routine "end_study"-------
continueRoutine = True
routineTimer.add(3.000000)
# update component parameters for each repeat
# keep track of which components have finished
end_studyComponents = [study_over]
for thisComponent in end_studyComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
end_studyClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end_study"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = end_studyClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=end_studyClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *study_over* updates
    if study_over.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
        # keep track of start time/frame for later
        study_over.frameNStart = frameN  # exact frame index
        study_over.tStart = t  # local t and not account for scr refresh
        study_over.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(study_over, 'tStartRefresh')  # time at next scr refresh
        study_over.setAutoDraw(True)
    if study_over.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > study_over.tStartRefresh + 3-frameTolerance:
            # keep track of stop time/frame for later
            study_over.tStop = t  # not accounting for scr refresh
            study_over.frameNStop = frameN  # exact frame index
            win.timeOnFlip(study_over, 'tStopRefresh')  # time at next scr refresh
            study_over.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in end_studyComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end_study"-------
for thisComponent in end_studyComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
eyetracker.setConnectionState(False)
ioServer.quit()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
