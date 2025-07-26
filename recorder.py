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
DECIBLE_METER_MIN_DB, DECIBLE_METER_MAX_DB = 0, 90
DECIBLE_METER_BAR_WIDTH = 40
QUIET_ROOM_DECIBLES = 30
NORMAL_CONVERSATION_DECIBLES = 55
TALKING_DIRECTLY_INTO_MIC_DECIBLES = 75
MIN_DECIBLES_BEFORE_SCALING_OFFSET = 20

CALIBRATION_INTRO_MESSAGE = f"""
We do some math to convert microphone input into decibles.
However, the result needs to be calibrated to be accurate.

How many decibles should we add to the calculated value to 
determine the decibles of the sound that is hitting your mic?

When you are ready, we will begin recording and we'll output
the calculated decible value.  Determine how many decibles
to add to this output to achieve the following:

   - Quiet Room: {QUIET_ROOM_DECIBLES} dB
   - Normal Conversation: {NORMAL_CONVERSATION_DECIBLES} dB
   - Talking Directly Into Mic: {TALKING_DIRECTLY_INTO_MIC_DECIBLES} dB

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

        # Calculate decibles.
        decibles = _calculate_decibles(recorded_input_data, offset_to_computed_decibles)

        # Output computed decibles to user.
        sys.stdout.write("\r" + _render_decible_meter(round(decibles)))
        sys.stdout.flush()

    # Cleanup
    pyaudio_input_stream.stop_stream()
    pyaudio_input_stream.close()
    pyaudio_instance.terminate()

def listen_for_and_transcribe_potential_wake_word(
        offset_to_computed_decibles,
        wake_word_max_length_in_seconds=1.5,
        decibles_that_indicate_speech=50):

    # Check offset_to_computed_decibles
    if not isinstance(offset_to_computed_decibles, (int, float)):
        raise TypeError(f"'offset_to_computed_decibles' must be a number, got {type(offset_to_computed_decibles).__name__!r}")

    if (offset_to_computed_decibles <= 0):
        raise ValueError(f"'offset_to_computed_decibles' must be >= 0, got {x}.  Microphone was not calibrated.")

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

    # In a loop, wait until input data exceeds some threshold decibles, then record.
    # Record until we hit some threshold seconds that cover the longest possible time
    # that the wake word is uttered.  Then perform speach-to-text and yield the text.
    is_recording = False
    recorded_seconds = 0
    buffered_input_data = 0
    recorded_frames = []
    
    try:
        while True:
            # Read from the input/mic stream.  
            # Will wait until buffer is full before returning.
            prior_buffered_input_data = buffered_input_data
            buffered_input_data = pyaudio_input_stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            recorded_seconds += SECONDS_IN_BUFFER

            # Calculate decibles of audio that was in the buffer.
            decibles = _calculate_decibles(buffered_input_data, offset_to_computed_decibles):
            print(f"decibles is {decibles}")

            # If we are recording then determine if we are done recording.
            if is_recording:

                # Append buffered input to recorded frames.
                recorded_frames.append(buffered_input_data)

                # Have we been recording for enough time?
                if (recorded_seconds >= wake_word_max_length_in_seconds):
                    
                    # Stop recording
                    is_recording = False
                    
                    # Perform in-memory speach-to-text
                    pcm = b''.join(recorded_frames)
                    recording = np.frombuffer(pcm, dtype=np.int16)
                    recording = recording.astype(np.float32) / 32768.0
                    transcription  = WHISPER_MODEL.transcribe(recording, fp16=False, temperature=[0.0], compression_ratio_threshold=None, logprob_threshold=None)

                    # Yield the transcription text and clear the recorded frames.
                    yield transcription["text"].strip()
                    recorded_frames = []
            elif (decibles > decibles_that_indicate_speech):
                # Speech detected; start recording.
                is_recording = True

                # The prior buffer might contain some speech we need, 
                # so record it along with the current buffer.
                recorded_frames.append(prior_buffered_input_data)
                recorded_frames.append(buffered_input_data)
                recorded_seconds = 2 * SECONDS_IN_BUFFER

                print("recording")
            else:
                # We aren't recording and shouldn't start.
                recorded_seconds = 0
    finally:
        # Cleanup when generator closes
        pyaudio_input_stream.stop_stream()
        pyaudio_input_stream.close()
        pyaudio_instance.terminate()

def _render_decible_meter(decibles):
    # clamp
    x = max(DECIBLE_METER_MIN_DB, min(DECIBLE_METER_MAX_DB, decibles))

    # fill ratio
    f = (x - DECIBLE_METER_MIN_DB) / (DECIBLE_METER_MAX_DB - DECIBLE_METER_MIN_DB)
    filled = int(f * DECIBLE_METER_BAR_WIDTH)
    bar = "█" * filled + "─" * (DECIBLE_METER_BAR_WIDTH - filled)
    return f"[{bar}] {decibles:5.1f} dB"

def _calculate_decibles(recorded_input_data, offset_to_computed_decibles):
    # Calculate decibles of recording to quantify the loudness
    rms_of_recorded_input_data = audioop.rms(recorded_input_data, 2)
    decibles = 20 * math.log10(rms_of_recorded_input_data / 32768.0)
    decibles_with_offset = decibles + offset_to_computed_decibles

    # Fixed offset doesn't honor how decibles scale as sound gets louder.
    # After a minimum decible value (with offset), scale the offset before applying it.
    if (decibles_with_offset <= MIN_DECIBLES_BEFORE_SCALING_OFFSET):
        decibles = decibles_with_offset
    else:
        offset_scale = (TALKING_DIRECTLY_INTO_MIC_DECIBLES - QUIET_ROOM_DECIBLES) / (offset_to_computed_decibles - QUIET_ROOM_DECIBLES)
        decibles = (decibles * offset_scale) + TALKING_DIRECTLY_INTO_MIC_DECIBLES

    return decibles
