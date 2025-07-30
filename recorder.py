import pyaudio
import json
import queue
from vosk import Model, KaldiRecognizer, SetLogLevel
import math
from pathlib import Path
import click

# Constants
FRAMES_PER_SECOND = 16000 # 16000 Hz
FRAMES_PER_BUFFER = 2048  # 2048 / 16000 Hz  =  128ms @ 16kHz microphone read
MAX_INPUT_QUEUE_SIZE_IN_SECONDS = 2
MAX_INPUT_QUEUE_SIZE = round(MAX_INPUT_QUEUE_SIZE_IN_SECONDS / (FRAMES_PER_BUFFER / FRAMES_PER_SECOND))

WAKE_WORD_SAMPLES = 10
WAKE_WORD_SAMPLE_MAX_ALTERNATIVES = 5
WAKE_WORDS_JSON_FILE_NAME = "wake_words.json"
ESTABLISH_WAKE_WORDS_INTRO_MESSAGE = f"""
Let's establish your wake word phrase.
Please speak your wake word phrase and then pause for the system 
to transcribe it (the transcription will be printed to the screen).
Then repeat your wake word phrase again.  

Keep repeating it until all {WAKE_WORD_SAMPLES} samples are collected.

For each sample, I will tell you my {WAKE_WORD_SAMPLE_MAX_ALTERNATIVES} best guesses
as to what you said.

The samples will be written to: {WAKE_WORDS_JSON_FILE_NAME}.

Press any key to start…
"""

MODEL_PATH   = "vosk-model-small-en-us-0.15"   # any model works
SetLogLevel(-1) # Silence vosk logs, must be before loading the model
MODEL = Model(MODEL_PATH)


import time
import numpy
import audioop
import re

SECONDS_IN_BUFFER = FRAMES_PER_BUFFER / FRAMES_PER_SECOND # 0.125 seconds
DECIBLE_METER_MIN_DB, DECIBLE_METER_MAX_DB = 0, 90

def establish_wake_words():

    # Create the recognizer.  The grammar argument is optional and we omit it
    # to instruct Vosk to switch to it's full language model.  An open vocabulary
    # gives us discovery: we learn how the model actually hears us.
    kaldi_recognizer = KaldiRecognizer(MODEL, FRAMES_PER_SECOND)

    # We ask for X best guesses so we can maximize the value of the sample.
    kaldi_recognizer.SetMaxAlternatives(WAKE_WORD_SAMPLE_MAX_ALTERNATIVES)

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

    # Loop until all samples are collected.
    sampled_wake_words = []
    sample_number = 0
    
    try:
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
                sample_number += 1

                # We could average the confidence values from each word and then compare
                # against some threshold before accepting the word.  
                # If so we have to add this: kaldi_recognizer.SetWords(True)
                # Potential future improvement.
                recognizer_result_json = kaldi_recognizer.Result()
                recognizer_result_dictionary = json.loads(recognizer_result_json)

                for alternatives in recognizer_result_dictionary.get("alternatives", []):
                    phrase = alternatives.get("text", "").strip().lower()

                    # Ignore silence.
                    if phrase:          
                        sampled_wake_words.append(phrase)
                        print(f"({sample_number}): \"{phrase}\"")

                print()
                
                if (sample_number >= WAKE_WORD_SAMPLES):
                    break
    finally:
        # Close out the input stream
        pyaudio_input_stream.stop_stream()
        pyaudio_input_stream.close()
        pyaudio_instance.terminate()

    # Load existing file (or start empty)
    wake_words_json_file_path = Path(WAKE_WORDS_JSON_FILE_NAME)
    if wake_words_json_file_path.exists():
        saved_wake_words = json.loads(wake_words_json_file_path.read_text())
    else:
        saved_wake_words = []

    # De-dupe, remove empty string and sort
    wake_words_to_save = set(saved_wake_words + sampled_wake_words)
    wake_words_to_save.discard("")
    wake_words_to_save = sorted(wake_words_to_save)

    # Overwrite with the updated list
    wake_words_json_file_path.write_text(
        json.dumps(wake_words_to_save, indent=2),
        encoding="utf-8")

    print(f"Captured all samples!  See: {wake_words_json_file_path}")

def wait_for_wake_words():

    # Read wake words from JSON file into a set
    wake_words_json_file_path = Path(WAKE_WORDS_JSON_FILE_NAME)
    
    if not wake_words_json_file_path.exists():
        raise ValueError(f"Wake words haven't been established.  Can't find file: {WAKE_WORDS_JSON_FILE_NAME}.")

    with wake_words_json_file_path.open("r", encoding="utf-8") as f:
        wake_words_list = json.load(f)
    
    # The "unknown‐word" escape hatch that Kaldi/Vosk keeps in every language‑model.
    # Vosk prunes the decoder so it can only emit the tokens in the grammar.
    # If you leave the list at just your wake‑words, the recogniser is forced to pick 
    # whichever phrase is closest to any incoming speech even if the user said 
    # something totally different.
    wake_words_list.add("[unk]") 

    # Create the recognizer.
    grammar_json = wake_words_json = json.dumps(wake_words_list)
    kaldi_recognizer = KaldiRecognizer(MODEL, FRAMES_PER_SECOND, wake_words_json)


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