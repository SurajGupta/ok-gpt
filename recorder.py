import pyaudio
import json
import sys
import queue
from vosk import Model, KaldiRecognizer
import math

# Constants
MODEL_PATH   = "vosk-model-small-en-us-0.15"   # any model works
MODEL = Model(MODEL_PATH)
FRAMES_PER_SECOND = 16000 # 16000 Hz
FRAMES_PER_BUFFER = 2048  # 2048 / 16000 Hz  =  128ms @ 16kHz microphone read
MAX_INPUT_QUEUE_SIZE_IN_SECONDS = 2
WAKE_WORD_SAMPLES = 10
WAKE_WORDS_JSON_FILE_NAME = "wake_words.json"

ESTABLISH_WAKE_WORDS_INTRO_MESSAGE = f"""
Let's establish your wake word phrase.
Please speak your wake word phrase and then pause for the system 
to transcribe it (the transcription will be printed to the screen).
Then repeat your wake word phrase again.  

Keep repeating it until all {WAKE_WORD_SAMPLES} samples are collected.

The samples will be written to:  {WAKE_WORDS_JSON_FILE_NAME}.

Press any key to start…
"""

# Calculations
MAX_INPUT_QUEUE_SIZE = round(MAX_INPUT_QUEUE_SIZE_IN_SECONDS / (FRAMES_PER_BUFFER / FRAMES_PER_SECOND))



import time
import numpy
import audioop
import click
from pathlib import Path
import re



SECONDS_IN_BUFFER = FRAMES_PER_BUFFER / FRAMES_PER_SECOND # 0.125 seconds
DECIBLE_METER_MIN_DB, DECIBLE_METER_MAX_DB = 0, 90
DECIBLE_METER_BAR_WIDTH = 40
QUIET_ROOM_DECIBLES = 30
NORMAL_CONVERSATION_DECIBLES = 55
TALKING_DIRECTLY_INTO_MIC_DECIBLES = 75
MIN_DECIBLES_BEFORE_SCALING_OFFSET = 20



def establish_wake_words():

    # Create the recognizer.  The grammar argument is optional and we omit it
    # to instruct Vosk to switch to it's full language model.  An open vocabulary
    # gives us discovery: we learn how the model actually hears us.
    kaldi_recognizer = KaldiRecognizer(MODEL, FRAMES_PER_SECOND)

    # We ask for the confidence score to be included with the transcription.
    # The confidence is specified per word, not per phrase and a start/end time
    # is also included per word.
    # The confidence will be in the range 0 → 1 (close to 1 ≈ high certainty).
    # Values are posterior probabilities computed by Kaldi’s decoder; they are 
    # not strictly calibrated but are useful for relative filtering 
    # (e.g. discard words whose conf < 0.3).  Output will look something like this:
    # {
    #     "text": "hey gizmo",
    #     "result": [
    #         { "word": "hey",   "start": 0.12, "end": 0.40, "conf": 0.87 },
    #         { "word": "gizmo", "start": 0.41, "end": 0.89, "conf": 0.75 }
    #     ]
    # }
    kaldi_recognizer.SetWords(True)

    # Instructions to user.
    click.pause(ESTABLISH_WAKE_WORDS_INTRO_MESSAGE)

    # Start listening and recording
    audio_queue = queue.Queue(maxsize=MAX_INPUT_QUEUE_SIZE)
    def _pyaudio_mic_callback(in_data, frame_count, time_info, status_flags):
        try:
            # Immediately raises queue.Full if no slot is free.
            audio_queue.put_nowait(in_data)
        except queue.Full:
            # Drop audio if queue is full
            pass

        # Tell PortAudio to keep recording
        return (None, pyaudio.paContinue)

    pyaudio_instance = pyaudio.PyAudio()
    pyaudio_input_stream = pyaudio_instance.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=FRAMES_PER_SECOND,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER,
        stream_callback = _pyaudio_mic_callback,
        start=True,
    )

    # Loop until 
    samples = 0
    while True:
        pcm = audio_queue.get()

        # AcceptWaveform calls Vosk’s endpoint detector on every chunk you feed it.
        # The method returns True (utterance is finished) when one of Kaldi’s built‑in
        # silence rules fires.  The rules evaluate the length of silence, the
        # length of the speech and model output to determine when the utterance is finished.
        # For wake words (short speech), rule 2 (500ms of silence) wins most often.
        # Note that for long silences (5s), the recognizer will trigger an endpoint.
        # So AcceptWaveform does not return False indefinitately.  While the endpoint
        # detection can be customized, we just use the default config.
        if kaldi_recognizer.AcceptWaveform(pcm): 
            recognizer_result = kaldi_recognizer.Result()
            wake_word = json.loads(recognizer_result)["text"].strip().lower()

            # ignore silence
            if wake_word:                          
                samples += 1
                print(f"{wake_word}")
                print recognizer_result

            if (samples > WAKE_WORD_SAMPLES):
                break

    # Close out the input stream
    pyaudio_input_stream.stop_stream()
    pyaudio_input_stream.close()
    pyaudio_instance.terminate()

    # sampled_wake_words = []

    # # Create a generator that yields potential wake words
    # wake_words_generator = _listen_for_and_transcribe_potential_wake_words(
    #     offset_to_computed_decibles,
    #     wake_word_max_length_in_seconds,
    #     decibles_that_indicate_speech,
    #     verbose = True, 
    #     print_sample_number_when_verbose= True)

    # try:
    #     # Iterate thru the required number of samples and save the spoken wake word phrases
    #     for i in range(WAKE_WORD_SAMPLES):
    #         # This will block until listen_for_and_transcribe_potential_wake_words yields a phrase
    #         phrase = next(wake_words_generator)  
    #         sampled_wake_words.append(phrase)
    # finally:
    #     # Now we tear down the generator (runs its finally:)
    #     wake_words_generator.close()

    # # Load existing file (or start empty)
    # wake_words_json_file_path = Path(WAKE_WORDS_JSON_FILE_NAME)
    # if wake_words_json_file_path.exists():
    #     saved_wake_words = json.loads(wake_words_json_file_path.read_text())
    # else:
    #     saved_wake_words = []

    # # Clean *each* phrase, de-dupe, and sort
    # wake_words_to_save =  { _clean_wake_word_phrase(w) for w in saved_wake_words + sampled_wake_words }

    # # Remove empty string and sort
    # wake_words_to_save.discard("")
    # wake_words_to_save = sorted(wake_words_to_save)

    # # Overwrite with the updated list
    # wake_words_json_file_path.write_text(
    #     json.dumps(wake_words_to_save, indent=2),
    #     encoding="utf-8")

    # print(f"Captured all samples!  See: {wake_words_json_file_path}")

def wait_for_wake_words(
        offset_to_computed_decibles,
        wake_word_max_length_in_seconds=1.5,
        decibles_that_indicate_speech=50,
        verbose=False):
    _check_offset_to_computed_decibles(offset_to_computed_decibles)

    # Read wake words from JSON file into a set
    wake_words_json_file_path = Path(WAKE_WORDS_JSON_FILE_NAME)
    
    if not wake_words_json_file_path.exists():
        raise ValueError(f"Wake words haven't been established.  Can't find file: {WAKE_WORDS_JSON_FILE_NAME}.")

    with wake_words_json_file_path.open("r", encoding="utf-8") as f:
        wake_words_list = json.load(f)

    wake_words_set = set(wake_words_list)

    # Create a generator that yields potential wake words
    wake_words_generator = _listen_for_and_transcribe_potential_wake_words(
        offset_to_computed_decibles,
        wake_word_max_length_in_seconds,
        decibles_that_indicate_speech,
        verbose = verbose, 
        print_sample_number_when_verbose = False)

    try:
        # Loop until the a wake word is uttered
        while True:
            # This will block until listen_for_and_transcribe_potential_wake_words yields a phrase
            possible_wake_word = next(wake_words_generator)

            possible_wake_word = _clean_wake_word_phrase(possible_wake_word)

            if (possible_wake_word in wake_words_set):
                break
    finally:
        # Now we tear down the generator (runs its finally:)
        wake_words_generator.close()

def _listen_for_and_transcribe_potential_wake_words(
        offset_to_computed_decibles,
        wake_word_max_length_in_seconds,
        decibles_that_indicate_speech,
        verbose,
        print_sample_number_when_verbose):

    _check_offset_to_computed_decibles(offset_to_computed_decibles)

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
    buffered_input_data = b''
    recorded_frames = []
    recorded_text = ""
    sample_number = 0
    
    try:
        while True:
            # Read from the input/mic stream.  
            # Will wait until buffer is full before returning.
            prior_buffered_input_data = buffered_input_data
            buffered_input_data = pyaudio_input_stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            recorded_seconds += SECONDS_IN_BUFFER

            # Calculate decibles of audio that was in the buffer.
            decibles = _calculate_decibles(buffered_input_data, offset_to_computed_decibles)
            
            # Output computed decibles, recording status, and last transcription.
            if verbose:
                _write_transcription_verbose_output(decibles, is_recording, print_sample_number_when_verbose, sample_number, recorded_text)

            # If we are recording then determine if we are done recording.
            if is_recording:

                # Append buffered input to recorded frames.
                recorded_frames.append(buffered_input_data)

                # Have we been recording for enough time?
                if (recorded_seconds >= wake_word_max_length_in_seconds):
                    
                    # Stop recording
                    is_recording = False
                    if verbose and print_sample_number_when_verbose:
                        sample_number += 1

                    # Perform in-memory speach-to-text
                    pcm = b''.join(recorded_frames)
                    recording = numpy.frombuffer(pcm, dtype=numpy.int16)
                    recording = recording.astype(numpy.float32) / 32768.0
                    transcription  = WHISPER_MODEL.transcribe(recording, fp16=False, temperature=[0.0], compression_ratio_threshold=None, logprob_threshold=None)

                    # Yield the transcription text and clear the recorded frames.
                    recorded_text = transcription["text"].strip()
                    yield recorded_text
                    recorded_frames = []

            elif (decibles > decibles_that_indicate_speech):
                # Speech detected; start recording.
                is_recording = True

                # The prior buffer might contain some speech we need, 
                # so record it along with the current buffer.
                recorded_frames.append(prior_buffered_input_data)
                recorded_frames.append(buffered_input_data)
                recorded_seconds = 2 * SECONDS_IN_BUFFER
                recorded_text = ""
                # print("recording started")
            else:
                # We aren't recording and shouldn't start.
                recorded_seconds = 0
    finally:
        # Write the final transcription
        if verbose:
            _write_transcription_verbose_output(decibles, is_recording, print_sample_number_when_verbose, sample_number, recorded_text)

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

def _write_transcription_verbose_output(decibles, is_recording, print_sample_number_when_verbose, sample_number, recorded_text):
    decible_meter = _render_decible_meter(round(decibles))

    sys.stdout.write("\033[2F")
    sys.stdout.write("\r\033[K" + decible_meter + "\n")

    if is_recording:
        recording_state = (" " * int(((DECIBLE_METER_BAR_WIDTH - 13)/2))) + "<< recording >>"
    elif recorded_text == "":
        recording_state = ""
    else:
        recording_state = " "
        if print_sample_number_when_verbose:
            recording_state = recording_state + f"({sample_number}): "
        recording_state = recording_state + " \"" + recorded_text + "\""

    sys.stdout.write("\r\033[K" + recording_state + "\n")
    sys.stdout.flush()

def _clean_wake_word_phrase(s: str) -> str:
    # keep only alphanumeric or space
    cleaned = "".join(c for c in s if c.isalnum() or c == " ")

    # lower case
    cleaned = cleaned.lower()

    # collapse runs of spaces
    cleaned = " ".join(cleaned.split())

    # trim
    cleaned = cleaned.strip()

    return cleaned
