import audioop
import whisper
import pyaudio
import wave
import os
import numpy as np
import time
import tempfile

ambient_detected = False
rms_that_indicates_speech = 500

WHISPER_MODEL = whisper.load_model("tiny.en")

FRAMES_PER_SECOND = 16000 # 16000 Hz
FRAMES_PER_BUFFER = 2000  # 2000 / 16000 Hz  =  125ms @ 16kHz microphone read

def live_speech(wake_word_max_length_in_seconds=2):
    global ambient_detected
    global rms_that_indicates_speech

    seconds_per_buffer = FRAMES_PER_BUFFER/FRAMES_PER_SECOND

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

    while True:
        # Read from the input/mic stream.  
        # Will wait until buffer is full before returning.
        recording = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
        recorded_seconds += seconds_per_buffer

        # Calculate RMS of recording to quantify the loudness
        rms_of_recording = audioop.rms(recording, 2)

        print(f"RMS is {rms_of_recording}")

        # At startup, determine the ambient sound level and use that to determine
        # what the level will be considered speech
        if not ambient_detected:
            if recorded_seconds < 1:
                if recorded_seconds == seconds_per_buffer:
                    print("Detecting ambient noise...")
                else:
                    if rms_that_indicates_speech < rms_of_recording:
                        rms_that_indicates_speech = rms_of_recording
                continue
            elif recorded_seconds == 1:
                print("Listening...")
                rms_that_indicates_speech = rms_that_indicates_speech * 4
                print(f"RMS that indicates speech is {rms_that_indicates_speech}")
                ambient_detected = True
                continue

        if is_recording:
            frames.append(recording)
            if (recorded_seconds >= wake_word_max_length_in_seconds):
                is_recording = False
                
                start = time.time()
                pcm = b''.join(frames)
                with tempfile.TemporaryFile(suffix=".wav") as tmp:
                    with wave.open(tmp, "wb") as wf:
                        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                        wf.writeframes(pcm)
                    tmp.seek(0)                     # rewind so ffmpeg can read
                result = WHISPER_MODEL.transcribe(tmp.name, fp16=False)
                end = time.time()
                length = end - start
                print("It took", length, "seconds!")
                print(result["text"].strip())

                start = time.time()
                pcm = b''.join(frames)
                the_audio = np.frombuffer(pcm, dtype=np.int16)
                the_audio = the_audio.astype(np.float32) / 32768.0
                result  = WHISPER_MODEL.transcribe(the_audio, fp16=False)
                end = time.time()
                length = end - start
                print("It took", length, "seconds!")
                print(result["text"].strip())

                start = time.time()
                wf = wave.open("audio.wav", 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(FRAMES_PER_SECOND)
                wf.writeframes(b''.join(frames))
                wf.close()
                result = WHISPER_MODEL.transcribe(
                    "audio.wav",
                    fp16=False
                )
                end = time.time()
                length = end - start
                print("It took", length, "seconds!")

                print(result["text"].strip())

                os.remove("audio.wav")

                yield result["text"].strip()
                frames = []
        elif (rms_of_recording > rms_that_indicates_speech):
            is_recording = True
            recorded_seconds = seconds_per_buffer
            print("recording")
        else:
            recorded_seconds = 0

    # TODO: do these when breaking from generator
    stream.stop_stream()
    stream.close()
    audio.terminate()
