import os

import assemblyai as aai
from dotenv import load_dotenv
from elevenlabs import generate, stream
from openai import OpenAI


class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = os.getenv(AAI_API_KEY)
        self.elevenlabs_api_key = os.getenv(ELEVENLABS_API_KEY)
        self.openai_client = OpenAI(api_key=os.getenv(OPENAI_API_KEY))

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

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            # print(transcript.text, end="\r\n")
            # send to a new method
            self.generate_ai_response(transcript)

        else:
            print(transcript.text, end="\r")

    def on_error(self, error: aai.RealtimeError):
        # print("An error occured:", error)
        return

    def on_close(self):
        # print("Closing Session")
        return
