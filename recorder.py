import audioop
import whisper
import pyaudio
import wave
import os

whisper_model = whisper.load_model("base")
ambient_detected = False
speech_volume = 50

def live_speech(wait_time=10):
    global ambient_detected
    global speech_volume

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=self.pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=4000,
        start=True,                        # start right away
    )

    frames = []
    recording = False
    frames_recorded = 0

    while True:
        frames_recorded += 1
        data = stream.read(4000)
        rms = audioop.rms(data, 2)

        if not ambient_detected:
            if frames_recorded < 40:
                if frames_recorded == 1:
                    print("Detecting ambient noise...")
                if frames_recorded > 5:
                    if speech_volume < rms:
                        speech_volume = rms
                continue
            elif frames_recorded == 40:
                print("Listening...")
                speech_volume = speech_volume * 3
                ambient_detected = True

        if rms > speech_volume:
            recording = True
            frames_recorded = 0
        elif recording and frames_recorded > wait_time:
            recording = False

            wf = wave.open("audio.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            result = whisper_model.transcribe(
                "audio.wav",
                fp16=False
            )

            os.remove("audio.wav")

            yield result["text"].strip()

            frames = []

        if recording:
            frames.append(data)

    # TODO: do these when breaking from generator
    stream.stop_stream()
    stream.close()
    audio.terminate()
