from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    groq,
    noise_cancellation,
    silero,
)
from client import TTS
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
CHATTERBOX_SERVER_URL = "ws://46.18.108.33:8765"

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful voice AI assistant. which answer in one line to every question..")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=groq.STT(
                model="whisper-large-v3-turbo",
                language="en",
            ),
        llm=groq.LLM(
            model="llama3-8b-8192"
        ),
        tts=TTS(websocket_url=CHATTERBOX_SERVER_URL,
        chunk_size= 50,
        fade_duration = 0.06,
        context_window = 50),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        # room_input_options=RoomInputOptions(
        #     # LiveKit Cloud enhanced noise cancellation
        #     # - If self-hosting, omit this parameter
        #     # - For telephony applications, use `BVCTelephony` for best results
        #     noise_cancellation=noise_cancellation.BVC(), 
        # ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))