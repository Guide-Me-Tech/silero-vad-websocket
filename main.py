import os
import asyncio
import threading
import time
import logging
from typing import List
import numpy as np
from livekit import api
from livekit.api import LiveKitAPI
from livekit import rtc
from silero_vad import read_audio, get_speech_timestamps
import json
from time import perf_counter
import subprocess
import traceback
import warnings
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
# logging.disable(logging.debug)
logger = logging.getLogger(__name__)

# Create handlers
file_handler = logging.FileHandler("bot_instance.log")
stdout_handler = logging.StreamHandler()

# Set level for handlers
file_handler.setLevel(logging.INFO)
stdout_handler.setLevel(logging.INFO)

# Create formatters and add them to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

warnings.filterwarnings("ignore")

# dotenv.load_dotenv()
configs = CONFIG


from .utils_vad import init_jit_model, OnnxWrapper
import torch
torch.set_num_threads(1)

def load_silero_vad(onnx=False):
    model_name = 'silero_vad.onnx' if onnx else 'silero_vad.jit'
    package_path = "silero_vad.data"
    
    try:
        import importlib_resources as impresources
        model_file_path = str(impresources.files(package_path).joinpath(model_name))
    except:
        from importlib import resources as impresources
        try:
            with impresources.path(package_path, model_name) as f:
                model_file_path = f
        except:
            model_file_path = str(impresources.files(package_path).joinpath(model_name))

    if onnx:
        model = OnnxWrapper(model_file_path, force_onnx_cpu=True)
    else:
        model = init_jit_model(model_file_path)
    
    return model

def memory_vad_check(
    raw_audio: bytes, sample_rate: int, vad_model, min_silence_ms: int = 500
) -> bool:
    """
    Returns True if speech is still ongoing.
    Returns False if we've reached at least `min_silence_ms` of silence (i.e. no more speech).
    Raises NoSpeechDetected if no speech is found at all.
    """

    # Convert raw PCM (16-bit, mono) to a NumPy array of floats in [-1, 1]
    # (Silero’s default usage can also accept int16, but normalizing is a bit safer)
    raw_audio_copy = raw_audio[:]
    audio_int16 = np.frombuffer(raw_audio_copy, dtype=np.int16)

    if len(audio_int16) == 0:
        raise NoSpeechDetected("Audio buffer is empty.")

    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # Get timestamps
    speech_timestamps = get_speech_timestamps(
        audio=audio_float32,
        model=vad_model,
        sampling_rate=sample_rate,  # If not specified, defaults to 16000 for some models
        min_silence_duration_ms=min_silence_ms,
        min_speech_duration_ms=500,
    )

    if not speech_timestamps:
        # No valid speech at all in this buffer
        del audio_int16
        del audio_float32
        del raw_audio_copy
        raise NoSpeechDetected()

    # By default, speech_timestamps is a list of dicts with "start" and "end" in samples
    audio_length = len(audio_float32)
    last_speech_end = speech_timestamps[-1]["end"]

    # If the waveform after last detected speech is greater than some threshold, treat it as silence.
    # By default, 700 ms @ 48kHz = 700 * 48 samples but you can adjust for your sample rate.
    # 700 ms at 48kHz = 0.7 * 48000 ~ 33600 samples
    threshold_samples = int(0.7 * sample_rate)

    if audio_length - last_speech_end >= threshold_samples:
        # Enough trailing silence → stop
        del audio_int16
        del audio_float32
        del raw_audio_copy
        return False
    del audio_int16
    del audio_float32
    del raw_audio_copy
    return True  # Still speech continuing


class NoSpeechDetected(Exception):
    def __init__(self, message="No Speech Detected"):
        self.message = message
        super().__init__(self.message)


async def HandleAudioTrack(audio_stream_input: rtc.AudioStream, audio_track, room_name, remote_participan_sid, remote_participat_name):
    logging.info(f"Starting to receive audio stream")
    vad_model = load_silero_vad(onnx=True)

    # Use a memory buffer for raw PCM data
    pcm_buffer = bytearray()
    counter_of_frames = 0
    check_interval = 100  # check VAD every 100 frame events
    # 1 frame event =  10 ms at 48 kHz

    # We assume the first frame in the iteration can give us sample_rate, num_channels
    sample_rate = None
    num_channels = None
    logging.info("Iterating over audio stream")
    # await asyncio.sleep(5)
    no_speech_counter = 0
    
    async for frame_event in audio_stream_input:

        buffer = frame_event.frame
        if sample_rate is None:
            sample_rate = buffer.sample_rate
        if num_channels is None:
            num_channels = buffer.num_channels
        # logging.info("Getting buffer")
        # Accumulate raw PCM data
        pcm_buffer.extend(buffer.data)
        counter_of_frames += 1

        # Periodically check for silence
        if (
            counter_of_frames % check_interval == 0
        ):  # this check is done every 100 frames or 1 second
            logging.info("Checking for silence")
            try:
                # Pass the entire in-memory buffer to the VAD checker
                ongoing_speech = memory_vad_check(
                    raw_audio=pcm_buffer,
                    sample_rate=sample_rate,
                    vad_model=vad_model,
                )
            except NoSpeechDetected:
                # logging.info("No speech detected. Clearing buffer and continuing.")
                pcm_buffer.clear()
                # You might decide here if you want to keep waiting or break.
                # We'll just continue for new audio.
                no_speech_counter += 1
                if no_speech_counter > 30 * check_interval:
                    logging.info("No speech detected for too long. Exiting.")
                    break
                continue

            if not ongoing_speech:
                # If we reached silence notify the user
                # smph.release()
                # logging.debug("SEMAPHORE RELEASED")
                asyncio.sleep(1.5)
                logging.info(
                    "notify the client"
                )
                # Clear the buffer and break out for this track
                pcm_buffer.clear()
                



async def room_handler(
    room: rtc.Room = None,
    loop: asyncio.AbstractEventLoop = None,
    idx: str = "XXXXXX",
):
    """
    This function handles events that occur in a given room, including track subscriptions.
    args[0] ---- idx --- id of the room
    args[1] ---- JobCall object
    """
    audio_stream = None
    logging.debug("ROOM should subscribe to track now")

    # get audio track
    audio_track = None
    remote_participant = None
    logging.info("room.remote_participants: %s", room.remote_participants)
    while audio_stream is None:
        for participant in room.remote_participants.values():
            if remote_participant is None:
                remote_participant = participant
                logging.info(
                    "remote_participant.track_publications: %s",
                    remote_participant.track_publications,
                )
                
                for track in participant.track_publications.values():
                    if track.kind == rtc.TrackKind.KIND_AUDIO:
                        audio_track = track.track
                        audio_stream = rtc.AudioStream(audio_track)
                        break
        logging.warning("Could not find audio track, retrying...")
        await asyncio.sleep(1)
    asyncio.ensure_future(
        HandleAudioTrack(
            audio_stream,
            audio_track.sid,
            room.name,
            remote_participant.sid,
            remote_participant.name,
        )
    )
    
    print("Handle Audio Track function initialized")

    # @room.on("track_subscribed")
    # def on_track_subscribed(
    #     track: rtc.Track,
    #     publication: rtc.RemoteTrackPublication,
    #     participant: rtc.RemoteParticipant,
    # ):
    #     if track.kind == rtc.TrackKind.KIND_AUDIO:

    #         print("Subscribed to an Audio Track")
    #         logger.debug("Track info: %s", track)
    #         audio_stream = rtc.AudioStream(track)

    #         asyncio.ensure_future(
    #             HandleAudioTrack(
    #                 audio_stream,
    #                 track.sid,
    #                 room.name,
    #                 participant.sid,
    #                 participant.name,
    #                 args[0],
    #                 args[1],
    #             )
    #         )
    #         print("Handle Audio Track function initialized")

    print("Participants in the room: %s", room.remote_participants)
    logger.debug("Tracks info: %s", room)

    @room.on("participant_disconnected")
    def participant_disconnected(participant: rtc.Participant):
        logger.warning("Participant disconnected: %s", participant)
        # You could raise an exception here if needed
        # raise ParticipantDisconnected("Participant disconnected")


def check_vad(filename, vad_model):
    """
    Returns True if the speech is ongoing, and False if speech has definitely stopped.
    Raises NoSpeechDetected if no speech is detected.
    """
    limit = 700
    try:
        wav = read_audio(filename)
        full_wav_length = len(wav)
        speech_timestamps = get_speech_timestamps(
            model=vad_model,
            audio=wav,
            min_silence_duration_ms=500,
            min_speech_duration_ms=500,
        )
        logger.debug("Speech timestamps: %s", speech_timestamps)

        if len(speech_timestamps) == 0:
            # no speech detected so keep listening
            raise NoSpeechDetected()

        last_speech_timestamp = speech_timestamps[-1]
        end_of_last_speech = last_speech_timestamp["end"]
    except NoSpeechDetected:
        raise NoSpeechDetected

    # the limit of the last speech timestamp
    # the limit is 300ms

    k = 16
    if full_wav_length - end_of_last_speech >= limit * k:
        # speech has stopped
        return False
    return True


class BotInstance:
    def __init__(self, idx):
        # configs = Configurations()
        self.idx = idx
        self.rooms = []
        self.participants = []

        self.livekit_client = LiveKitAPI(
            url=configs.livekit.url,
            api_key=configs.livekit.api_key,
            api_secret=configs.livekit.api_secret,
        )

    def HandleInbound(self):
        pass

    async def HandleOutBound(self, room: rtc.Room):
        raise NotImplementedError

    async def GetRooms(self):
        req = api.ListRoomsRequest()
        resp = await self.livekit_client.room.list_rooms(req)
        output = []
        for room in resp.rooms:
            if room.name.startswith(self.idx) and room.num_participants >= 2:
                output.append(room)
        # do some filtering
        return resp.rooms

    async def HandleInput(self):
        raise NotImplementedError


def generate_token(room_name: str):
    token = (
        api.AccessToken(
            api_key=configs.livekit.api_key, api_secret=configs.livekit.api_secret
        )
        .with_identity(f"bot - {room_name}")
        .with_name(f"bot - {room_name}")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
            )
        )
        .to_jwt()
    )
    print("Generated token for room: ", room_name)
    return token


SAMPLE_RATE = 48000
NUM_CHANNELS = 1


async def one_event_loop(idx: str, room_name):
    # configs = Configurations()
    loop = asyncio.get_event_loop()
    room = rtc.Room()

    connect_task = loop.create_task(
        room.connect(
            url=configs.livekit.url,
            token=generate_token(room_name),
        )
    )
    await connect_task

    # Schedule room handler concurrently
    listener_task = asyncio.create_task(room_handler(room, loop, idx))

    # Publish audio track
    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track_output = rtc.LocalAudioTrack.create_audio_track("sinewave", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE

    print("Publishing audio track to room: %s", room_name)
    publish_task = loop.create_task(
        room.local_participant.publish_track(track_output, options)
    )
    await publish_task
    logger.debug("Track published audio: %s", track_output)

    # Schedule the audio handler concurrently
    published_track_handler = asyncio.create_task(PublishedTrackHandler(source, idx))
    # wait_for_termination_message = asyncio.create_task(WaitForTermination(idx))
    # Ensure all tasks are running concurrently
    try:
        await asyncio.gather(listener_task, published_track_handler)
    except ParticipantDisconnected as e:
        logger.warning("Participant disconnected: %s", e)
        req = api.DeleteRoomRequest(room_name)
        await api.livekit_api.LiveKitAPI().room.delete_room(req)
        raise e


def run_in_thread(idx, room):
    print("Running one_event_loop in thread for room: %s", room)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(one_event_loop(idx, room))

    print("Event loop completed for room: %s, idx: %s", room, idx)
    try:
        loop.run_forever()
    except ParticipantDisconnected as e:
        logger.warning("Participant disconnected (thread): %s", e)
    except FinishThread as e:
        logger.warning(
            "Closing the thread because the assistant ended the conversation"
        )
    finally:
        print("Closing the loop for room: %s, idx: %s", room, idx)
        loop.close()
        raise FinishThread(message=idx)





def main():
    pass

if __name__ == "__main__":

    main()
