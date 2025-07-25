import whisper
import wave
import os
import numpy as np
import time

# Constants
WHISPER_MODEL = whisper.load_model("tiny.en")

import pyaudio
import audioop
import click
import math
import sys

FRAMES_PER_SECOND = 16000 # 16000 Hz
FRAMES_PER_BUFFER = 2000  # 2000 / 16000 Hz  =  125ms @ 16kHz microphone read
SECONDS_IN_BUFFER = FRAMES_PER_BUFFER / FRAMES_PER_SECOND # 0.125 seconds

CALIBRATION_TIME_IN_SECONDS = 30

CALIBRATION_INTRO_MESSAGE = f"""
We do some math to convert microphone input into decibles.
However, the result needs to be calibrated to be accurate.

How many decibles should we add to the calculated value to 
determine the decibles of the sound that is hitting your mic?

When you are ready, we will begin recording and we'll output
the calculated decible value.  Determine how many decibles
to add to this output to achieve the following:

   - Quiet Room: 30-35 dB
   - Soft Conversation: 55 db
   - Talking Directly Into Mic: 70-80 db

We'll stop after {CALIBRATION_TIME_IN_SECONDS} seconds.

Call this function again setting `offset_to_computed_decibles`
to your best guess.  Rinse-and-repeat until you've determine
the best offset.

Press any key to start…
"""

"\n\n\n\n\n\n"

def calibrate_decibles(offset_to_computed_decibles=0):
    
    # Instructions to user.
    click.pause(CALIBRATION_INTRO_MESSAGE)

    # Open the audio stream and start recording right away.
    pyaudio_instance = pyaudio.PyAudio()
    pyaudio_input_stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=FRAMES_PER_SECOND,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        start=True,
    )

    # Loop for CALIBRATION_TIME_IN_SECONDS
    elapsed_seconds = 0
    while (elapsed_seconds < CALIBRATION_TIME_IN_SECONDS):
        
        # Read from the input/mic stream, wait until buffer is full before returning.
        recorded_input_data = pyaudio_input_stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        elapsed_seconds += SECONDS_IN_BUFFER

        # Calculate decibles of recording to quantify the loudness
        rms_of_recorded_input_data = audioop.rms(recorded_input_data, 2)
        decibles = 20 * math.log10(rms_of_recorded_input_data / 32768.0)
        decibles_with_offset = decibles + offset_to_computed_decibles

        if (decibles_with_offset < 20):
            decibles = decibles_with_offset
        else:
            scale   = (80.0 - 35.0) / (offset_to_computed_decibles - 35.0)
            decibles = decibles * scale + 80.0

        # Output computed decibles to user.
        sys.stdout.write("\r" + render_meter(round(decibles)))
        sys.stdout.flush()
        # print(f"calculated decibles: {round(decibles)}")

    # Cleanup
    pyaudio_input_stream.stop_stream()
    pyaudio_input_stream.close()
    pyaudio_instance.terminate()

MIN_DB, MAX_DB = 0.0, 90.0
BAR_WIDTH = 40

def render_meter(decibles):
    # clamp
    x = max(MIN_DB, min(MAX_DB, decibles))

    # fill ratio
    f = (x - MIN_DB) / (MAX_DB - MIN_DB)
    filled = int(f * BAR_WIDTH)
    bar = "█" * filled + "─" * (BAR_WIDTH - filled)
    return f"[{bar}] {decibles:5.1f} dB"


def live_speech(wake_word_max_length_in_seconds=1.5):
    global ambient_detected
    global rms_that_indicates_speech

    SECONDS_IN_BUFFER = FRAMES_PER_BUFFER/FRAMES_PER_SECOND

    audio = pyaudio.PyAudio()

    # Open the audio stream and start recording right away.
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=FRAMES_PER_SECOND,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        start=True,
    )

    frames = []

    is_recording = False
    recorded_seconds = 0
    recording = 0

    while True:
        # Read from the input/mic stream.  
        # Will wait until buffer is full before returning.
        prior_recording = recording
        recording = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        recorded_seconds += SECONDS_IN_BUFFER

        # Calculate RMS of recording to quantify the loudness
        rms_of_recording = audioop.rms(recording, 2)

        # print(f"RMS is {rms_of_recording}")
        dBFS = 20 * math.log10(rms_of_recording / 32768.0)
        DB_OFFSET = 70
        dB   = dBFS + DB_OFFSET
    
        print(f"decibles is {dB}")

        db_that_indicates_speech = 55

        if is_recording:
            frames.append(recording)
            if (recorded_seconds >= wake_word_max_length_in_seconds):
                is_recording = False
                
                pcm = b''.join(frames)
                the_audio = np.frombuffer(pcm, dtype=np.int16)
                the_audio = the_audio.astype(np.float32) / 32768.0
                result  = WHISPER_MODEL.transcribe(the_audio, fp16=False, temperature=[0.0], compression_ratio_threshold=None, logprob_threshold=None)

                yield result["text"].strip()
                frames = []
        elif (dB > db_that_indicates_speech):
            is_recording = True
            frames.append(prior_recording)
            frames.append(recording)
            recorded_seconds = SECONDS_IN_BUFFER
            print("recording")
        else:
            recorded_seconds = 0

    # TODO: do these when breaking from generator
    stream.stop_stream()
    stream.close()
    audio.terminate()
