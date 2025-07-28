import whisper
import numpy
import pyaudio
import audioop
import click
import math
import sys
import json
from pathlib import Path
import re
import queue, time
from vosk import Model, KaldiRecognizer


# Constants
WHISPER_MODEL = whisper.load_model("tiny.en")
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
WAKE_WORD_SAMPLES = 10
WAKE_WORDS_JSON_FILE_NAME = "wake_words.json"

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

ESTABLISH_WAKE_WORDS_INTRO_MESSAGE = f"""
Let's establish your wake word phrase.
Please speak your wake word phrase and then pause for the system 
to transcribe it (the transcription will be printed to the screen).
Then repeat your wake word phrase again.  

Keep repeating it until all {WAKE_WORD_SAMPLES} samples are collected.

The samples will be written to:  {WAKE_WORDS_JSON_FILE_NAME}.

Press any key to start…
"""

WAKE_WORDS     = ["hey jellybot", "okay computer"]      # canonical list
SAMPLE_RATE    = 16000                               # model default
BLOCK_SIZE     = 6400                                # 0.4 s / 256 ms
MODEL_PATH     = "vosk-model-small-en-us-0.15"
model   = Model(MODEL_PATH)
grammar = json.dumps(WAKE_WORDS + ["[unk]"])
rec     = KaldiRecognizer(model, SAMPLE_RATE, grammar)
audio_q = queue.Queue(maxsize=10)


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

def establish_wake_words(
        offset_to_computed_decibles,
        wake_word_max_length_in_seconds=1.5,
        decibles_that_indicate_speech=50):
    _check_offset_to_computed_decibles(offset_to_computed_decibles)

    # Instructions to user.
    click.pause(ESTABLISH_WAKE_WORDS_INTRO_MESSAGE)
    
    sampled_wake_words = []

    # Create a generator that yields potential wake words
    wake_words_generator = _listen_for_and_transcribe_potential_wake_words(
        offset_to_computed_decibles,
        wake_word_max_length_in_seconds,
        decibles_that_indicate_speech,
        verbose = True, 
        print_sample_number_when_verbose= True)

    try:
        # Iterate thru the required number of samples and save the spoken wake word phrases
        for i in range(WAKE_WORD_SAMPLES):
            # This will block until listen_for_and_transcribe_potential_wake_words yields a phrase
            phrase = next(wake_words_generator)  
            sampled_wake_words.append(phrase)
    finally:
        # Now we tear down the generator (runs its finally:)
        wake_words_generator.close()

    # Load existing file (or start empty)
    wake_words_json_file_path = Path(WAKE_WORDS_JSON_FILE_NAME)
    if wake_words_json_file_path.exists():
        saved_wake_words = json.loads(wake_words_json_file_path.read_text())
    else:
        saved_wake_words = []

    # Clean *each* phrase, de-dupe, and sort
    wake_words_to_save =  { _clean_wake_word_phrase(w) for w in saved_wake_words + sampled_wake_words }

    # Remove empty string and sort
    wake_words_to_save.discard("")
    wake_words_to_save = sorted(wake_words_to_save)

    # Overwrite with the updated list
    wake_words_json_file_path.write_text(
        json.dumps(wake_words_to_save, indent=2),
        encoding="utf-8")

    print(f"Captured all samples!  See: {wake_words_json_file_path}")

def wait_for_wake_words(
        offset_to_computed_decibles,
        wake_word_max_length_in_seconds=1.5,
        decibles_that_indicate_speech=50,
        verbose=False):
    _check_offset_to_computed_decibles(offset_to_computed_decibles)

    pa     = pyaudio.PyAudio()
    stream = pa.open(
        format            = pyaudio.paInt16,
        channels          = 1,
        rate              = SAMPLE_RATE,
        input             = True,
        frames_per_buffer = BLOCK_SIZE,
        stream_callback   = _callback,
        start             = True,              # begin immediately
    )

    print("Listening for:", ", ".join(WAKE_WORDS))
    last_print = time.time()

    try:
        while stream.is_active():
            data = audio_q.get()               # blocks until next buffer

            # --- run recogniser on this block ------------------------------
            hotword_hit = rec.AcceptWaveform(data)

            # Low‑latency partials
            partial = json.loads(rec.PartialResult() or "{}").get("partial", "")
            if partial and time.time() - last_print > 0.5:
                print("partial:", partial.lower())
                last_print = time.time()

            if hotword_hit:
                text = json.loads(rec.Result())["text"].strip().lower()
                if text in WAKE_WORDS:
                    print(f"\n>>> WAKE WORD DETECTED: '{text}' <<<\n")
                    rec.Reset()                # clear state for next phrase
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()

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

def _check_offset_to_computed_decibles(offset_to_computed_decibles):
    if not isinstance(offset_to_computed_decibles, (int, float)):
        raise TypeError(f"'offset_to_computed_decibles' must be a number, got {type(offset_to_computed_decibles).__name__!r}")

    if (offset_to_computed_decibles <= 0):
        raise ValueError(f"'offset_to_computed_decibles' must be >= 0, got {x}.  Microphone was not calibrated.")

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

def _callback(in_data, frame_count, time_info, status_flags):
    if status_flags:
        # Drop-outs or input overflow get logged but we keep going
        print(status_flags, file=sys.stderr)
    try:
        audio_q.put_nowait(in_data)         # hand bytes to main thread
    except queue.Full:
        pass                                # main thread is busy -> skip
    return (None, pyaudio.paContinue)

