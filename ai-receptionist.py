import os

import assemblyai as aai
from dotenv import load_dotenv
from elevenlabs import generate, stream
from openai import OpenAI

load_dotenv()

aai_api_key = os.getenv("AAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = aai_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)

        self.transcriber = None

        # list of full transcripts by user & ai
        # with initial prompt
        self.full_transcript = [
            {
                "role": "system",
                "content": "You are a receptionist at a dental clinic. Be resourceful and efficient",
            }
        ]

    # real time transcription
    def start_transcription(self):
        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            end_utterance_silence_threshold=1000,
        )

        # connect mic & stream data to aai
        self.transcriber.connect()
        mic_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        self.transcriber.stream(mic_stream)

    def stop_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None

    # from aai documentation
    # https://www.assemblyai.com/docs/getting-started/transcribe-streaming-audio-from-a-microphone/python

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        # print("Session ID:", session_opened.session_id)
        return

    # define what to do with real time transcript
    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        print("DEBUG - Transcript received:", transcript.text)

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # print(transcript.text, end="\r\n")
            # send to a new method
            print("DEBUG - Final Transcript:", transcript.text)
            self.generate_ai_response(transcript)

        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        print("An error occured:", error)
        return

    def on_close(self):
        # print("Closing Session")
        return

    def generate_ai_response(self, transcript):
        # pause realtime transcription when communicating with openai api
        self.stop_transcription()
        self.full_transcript.append({"role": "user", "content": transcript.text})
        print(f"\nPatient:, {transcript.text}", end="\r\n")

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=self.full_transcript
        )

        ai_response = response.choices[0].message.content
        self.generate_audio(ai_response)
        self.start_transcription()

    # generate audio from elevenlabs
    def generate_audio(self, text):
        self.full_transcript.append({"role": "system", "content": text})
        print(f"\nAI Receptionist: {text}", end="\r\n")

        audio_stream = generate(
            text, self.elevenlabs_api_key, voice="Rachel", stream=True
        )

        stream(audio_stream)


greeting = "Welcome to the dental clinic. How can I help you today?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()
