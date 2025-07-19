import asyncio
import json
import websockets
from typing import AsyncGenerator, Optional, Dict, Any
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    StartInterruptionFrame,
    CancelFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.processors.frame_processor import FrameDirection


class ChatterboxWebSocketService(TTSService):
    """WebSocket-based Chatterbox TTS service for pipecat pipelines"""

    def __init__(
        self,
        websocket_url: str = "ws://localhost:8765",
        voice_prompt_path: Optional[str] = None,
        streaming_mode: bool = True,
        sample_rate: int = 24000,
        # Chatterbox-specific parameters
        chunk_size: int = 75,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        context_window: int = 70,
        fade_duration: float = 0.09,
        # Interruption handling
        reconnect_on_interrupt: bool = False,  # Whether to reconnect on interruption
        **kwargs
    ):
        # Initialize with required sample rate
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._websocket_url = websocket_url
        self._voice_prompt_path = voice_prompt_path
        self._streaming_mode = streaming_mode
        self._sample_rate = sample_rate
        
        # Chatterbox parameters
        self._chunk_size = chunk_size
        self._exaggeration = exaggeration
        self._temperature = temperature
        self._cfg_weight = cfg_weight
        self._context_window = context_window
        self._fade_duration = fade_duration
        
        # Interruption handling
        self._reconnect_on_interrupt = reconnect_on_interrupt
        
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._session_id: Optional[str] = None
        self._is_generating = False
        self._connection_lock = asyncio.Lock()
        self._started = False
        self._connect_timeout = 10.0
        self._max_retries = 3
        self._current_request_id = 0
        self._current_session_id = None  # Track current session for interruption
        
        logger.info(f"Chatterbox WebSocket service initialized for {websocket_url}")

    async def start(self, frame: Frame) -> None:
        """Start the service and establish WebSocket connection"""
        await super().start(frame)
        self._started = True
        
        # Try to establish connection with retries
        for attempt in range(self._max_retries):
            try:
                await self._ensure_connection()
                break
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == self._max_retries - 1:
                    logger.error("Failed to establish WebSocket connection after all retries")
                    raise
                await asyncio.sleep(1.0)

    async def stop(self, frame: Frame) -> None:
        """Stop the service and close WebSocket connection"""
        self._started = False
        await self._close_connection()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel any ongoing generation"""
        if self._is_generating:
            # Handle cancellation similar to interruption
            await self._stop_generation()
        await super().cancel(frame)
    
    async def _stop_generation(self):
        """Stop ongoing generation cleanly"""
        if not self._is_generating:
            return
            
        logger.info("Stopping ongoing TTS generation")
        self._is_generating = False
        
        # If in streaming mode and we have a session, send stream_end
        if self._streaming_mode and self._current_session_id and self._websocket and not self._websocket.closed:
            try:
                await self._websocket.send(json.dumps({
                    "type": "stream_end"
                }))
                logger.info(f"Sent stream_end for session {self._current_session_id}")
            except Exception as e:
                logger.error(f"Error sending stream_end: {e}")

    async def _ensure_connection(self):
        """Ensure WebSocket connection is established"""
        if not self._started:
            return
            
        async with self._connection_lock:
            if self._websocket is None or self._websocket.closed:
                await self._connect()

    async def _connect(self):
        """Establish WebSocket connection"""
        try:
            logger.info(f"Connecting to Chatterbox WebSocket server at {self._websocket_url}")
            
            self._websocket = await asyncio.wait_for(
                websockets.connect(
                    self._websocket_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=16 * 1024 * 1024  # 16MB max message size
                ),
                timeout=self._connect_timeout
            )
            
            logger.info("Chatterbox WebSocket connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Chatterbox WebSocket server: {e}")
            if self._websocket:
                try:
                    await self._websocket.close()
                except:
                    pass
            self._websocket = None
            raise

    async def _close_connection(self):
        """Close WebSocket connection"""
        async with self._connection_lock:
            if self._websocket:
                try:
                    await self._websocket.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket connection: {e}")
                finally:
                    self._websocket = None
                    self._session_id = None
                    logger.info("Chatterbox WebSocket connection closed")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text via WebSocket"""
        if not self._started:
            yield ErrorFrame("Chatterbox service not started")
            return
            
        try:
            await self._ensure_connection()
            
            if not self._websocket or self._websocket.closed:
                # Try to reconnect once
                try:
                    await self._connect()
                except Exception as e:
                    yield ErrorFrame(f"Chatterbox WebSocket connection failed: {str(e)}")
                    return

            logger.info(f"TTS for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Send TTS started frame
            yield TTSStartedFrame()
            
            self._is_generating = True
            
            try:
                if self._streaming_mode:
                    # Use streaming mode
                    async for frame in self._stream_synthesis(text):
                        yield frame
                else:
                    # Use non-streaming mode
                    async for frame in self._synthesize(text):
                        yield frame
                        
            except Exception as e:
                logger.error(f"Error during TTS generation: {e}")
                yield ErrorFrame(f"TTS generation failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in TTS generation: {e}")
            yield ErrorFrame(f"TTS generation failed: {str(e)}")
        finally:
            self._is_generating = False
            yield TTSStoppedFrame()

    async def _synthesize(self, text: str) -> AsyncGenerator[Frame, None]:
        """Non-streaming synthesis mode"""
        self._current_request_id += 1
        request_id = f"req_{self._current_request_id}"
        
        # Prepare synthesis request
        request = {
            "type": "synthesize",
            "text": text,
            "request_id": request_id,
            "params": {
                "chunk_size": self._chunk_size,
                "exaggeration": self._exaggeration,
                "temperature": self._temperature,
                "cfg_weight": self._cfg_weight,
                "context_window": self._context_window,
                "fade_duration": self._fade_duration
            }
        }
        
        if self._voice_prompt_path:
            request["params"]["voice_prompt_path"] = self._voice_prompt_path
            
        # Send request
        await self._websocket.send(json.dumps(request))
        
        # Receive response
        while self._is_generating:
            try:
                message = await asyncio.wait_for(self._websocket.recv(), timeout=30.0)
                
                # Parse JSON response
                data = json.loads(message)
                
                if data["type"] == "audio":
                    # Decode base64 audio
                    import base64
                    audio_bytes = base64.b64decode(data["audio_content"])
                    
                    yield TTSAudioRawFrame(
                        audio=audio_bytes,
                        sample_rate=data.get("sample_rate", self._sample_rate),
                        num_channels=data.get("channels", 1)
                    )
                    
                elif data["type"] == "synthesis_complete":
                    logger.info(f"Synthesis completed for request {request_id}")
                    break
                    
                elif data["type"] == "error":
                    logger.error(f"Synthesis error: {data.get('error')}")
                    yield ErrorFrame(f"Synthesis error: {data.get('error')}")
                    break
                    
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for synthesis response")
                yield ErrorFrame("Synthesis timeout")
                break
            except Exception as e:
                logger.error(f"Error receiving synthesis response: {e}")
                yield ErrorFrame(f"Synthesis error: {str(e)}")
                break

    async def _stream_synthesis(self, text: str) -> AsyncGenerator[Frame, None]:
        """Streaming synthesis mode - leveraging Chatterbox's native streaming"""
        session_id = f"session_{self._current_request_id}"
        self._current_request_id += 1
        self._current_session_id = session_id  # Store for interruption handling
        
        # Send stream start
        stream_start = {
            "type": "stream_start",
            "session_id": session_id,
            "params": {
                "chunk_size": self._chunk_size,
                "exaggeration": self._exaggeration,
                "temperature": self._temperature,
                "cfg_weight": self._cfg_weight,
                "context_window": self._context_window,
                "fade_duration": self._fade_duration
            }
        }
        
        if self._voice_prompt_path:
            stream_start["params"]["voice_prompt_path"] = self._voice_prompt_path
            
        await self._websocket.send(json.dumps(stream_start))
        
        # Wait for stream ready confirmation
        try:
            ready_msg = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
            ready_data = json.loads(ready_msg)
            
            if ready_data["type"] != "stream_ready":
                logger.error(f"Unexpected response: {ready_data}")
                yield ErrorFrame("Failed to start stream")
                return
                
            logger.info(f"Stream ready for session {session_id}")
            
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for stream ready")
            yield ErrorFrame("Stream initialization timeout")
            return
        
        # Send text
        await self._websocket.send(json.dumps({
            "type": "stream_text",
            "text": text
        }))
        
        # Send flush to trigger generation
        await self._websocket.send(json.dumps({
            "type": "stream_flush"
        }))
        
        # Receive streaming audio
        chunk_count = 0
        
        while self._is_generating:
            try:
                message = await asyncio.wait_for(self._websocket.recv(), timeout=0.5)
                
                # Check if it's a text message (JSON) or binary (audio)
                if isinstance(message, str):
                    # JSON message
                    data = json.loads(message)
                    
                    if data["type"] == "audio_chunk":
                        # Next message should be binary audio
                        chunk_index = data.get("chunk_index", chunk_count)
                        
                        # Receive the actual audio data (binary frame)
                        try:
                            audio_data = await asyncio.wait_for(self._websocket.recv(), timeout=1.0)
                            
                            if isinstance(audio_data, bytes):
                                yield TTSAudioRawFrame(
                                    audio=audio_data,
                                    sample_rate=data.get("sample_rate", self._sample_rate),
                                    num_channels=data.get("channels", 1)
                                )
                                chunk_count += 1
                                logger.debug(f"Received audio chunk {chunk_index}, size: {len(audio_data)} bytes")
                            else:
                                logger.warning(f"Expected binary audio, got: {type(audio_data)}")
                                
                        except asyncio.TimeoutError:
                            logger.warning("Timeout waiting for audio data after chunk header")
                            
                    elif data["type"] == "stream_complete":
                        logger.info(f"Stream completed for session {session_id}, {chunk_count} chunks")
                        break
                        
                    elif data["type"] == "error":
                        logger.error(f"Stream error: {data.get('error')}")
                        yield ErrorFrame(f"Stream error: {data.get('error')}")
                        break
                        
                elif isinstance(message, bytes):
                    # Direct binary audio (some Chatterbox configs might send this way)
                    yield TTSAudioRawFrame(
                        audio=message,
                        sample_rate=self._sample_rate,
                        num_channels=1
                    )
                    chunk_count += 1
                    
            except asyncio.TimeoutError:
                # Check if we're still supposed to be generating
                if not self._is_generating:
                    break
                # Small timeout is OK during streaming, continue
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed during streaming")
                break
            except Exception as e:
                logger.error(f"Error receiving stream data: {e}")
                yield ErrorFrame(f"Stream error: {str(e)}")
                break
        
        # Send stream end if still connected
        if self._websocket and not self._websocket.closed and self._is_generating:
            try:
                await self._websocket.send(json.dumps({
                    "type": "stream_end"
                }))
            except:
                pass

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        """Handle interruption when user speaks while TTS is playing"""
        await super()._handle_interruption(frame, direction)
        
        # Since Chatterbox doesn't have explicit interrupt messages like StyleTTS2,
        # we need to handle this differently
        if self._is_generating and self._websocket and not self._websocket.closed:
            logger.info("User interrupted - stopping Chatterbox generation")
            
            # Set flag to stop receiving audio
            self._is_generating = False
            
            # If in streaming mode, send stream_end to cleanly stop
            if self._streaming_mode and self._session_id:
                try:
                    await self._websocket.send(json.dumps({
                        "type": "stream_end"
                    }))
                    logger.info("Sent stream_end to stop generation")
                except Exception as e:
                    logger.error(f"Error sending stream_end: {e}")
            
            # Option 1: Close and reconnect for clean state (if enabled)
            if self._reconnect_on_interrupt:
                try:
                    await self._close_connection()
                    await asyncio.sleep(0.1)  # Brief pause
                    await self._ensure_connection()
                    logger.info("Reconnected after interruption")
                except Exception as e:
                    logger.error(f"Error during interruption reconnect: {e}")
            else:
                # Option 2: Just stop processing without reconnecting
                # Clear any session state
                self._current_session_id = None

    async def get_connection_status(self) -> dict:
        """Get current connection status"""
        if not self._websocket:
            return {"connected": False}
            
        if self._websocket.closed:
            return {"connected": False, "closed": True}
            
        return {
            "connected": True,
            "url": self._websocket_url,
            "is_generating": self._is_generating
        }


# Example usage and testing
async def test_chatterbox_websocket_tts():
    """Test function for the Chatterbox WebSocket TTS service"""
    service = ChatterboxWebSocketService(
        websocket_url="ws://localhost:8765",
        voice_prompt_path="/path/to/voice/reference.wav",  # Optional
        streaming_mode=True,  # Use native streaming
        chunk_size=75,
        exaggeration=0.5,
        temperature=0.8,
        cfg_weight=0.5,
        context_window=70,
        fade_duration=0.09,
        reconnect_on_interrupt=False  # Fast interruption without reconnect
    )
    
    try:
        # Create a proper frame
        start_frame = StartFrame()
        
        # Initialize service
        await service.start(start_frame)
        
        # Test connection status
        status = await service.get_connection_status()
        logger.info(f"Connection status: {status}")
        
        # Generate some test audio
        test_texts = [
            "Hello, this is a test of the Chatterbox WebSocket service.",
            "The native streaming capability provides incredibly low latency.",
            "This is perfect for real-time conversational AI applications."
        ]
        
        for test_text in test_texts:
            logger.info(f"Generating audio for: {test_text}")
            audio_frames = []
            chunk_times = []
            
            start_time = asyncio.get_event_loop().time()
            first_chunk_time = None
            
            async for frame in service.run_tts(test_text):
                current_time = asyncio.get_event_loop().time()
                
                if isinstance(frame, TTSAudioRawFrame):
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        logger.info(f"First chunk latency: {first_chunk_time - start_time:.3f}s")
                    
                    audio_frames.append(frame.audio)
                    chunk_times.append(current_time - start_time)
                    logger.debug(f"Received audio frame: {len(frame.audio)} bytes at {current_time - start_time:.3f}s")
                    
                elif isinstance(frame, TTSStartedFrame):
                    logger.info("TTS generation started")
                    
                elif isinstance(frame, TTSStoppedFrame):
                    total_time = asyncio.get_event_loop().time() - start_time
                    logger.info(f"TTS generation completed in {total_time:.3f}s")
                    
                elif isinstance(frame, ErrorFrame):
                    logger.error(f"TTS error: {frame.error}")
            
            if audio_frames:
                # Calculate statistics
                total_audio = sum(len(f) for f in audio_frames)
                audio_duration = total_audio / (2 * 24000)  # 16-bit audio at 24kHz
                real_time_factor = (chunk_times[-1] if chunk_times else 0) / audio_duration if audio_duration > 0 else 0
                
                logger.info(f"Generated {len(audio_frames)} chunks, {total_audio} bytes total")
                logger.info(f"Audio duration: {audio_duration:.2f}s, RTF: {real_time_factor:.3f}")
                
                # Save combined audio for testing
                combined_audio = b''.join(audio_frames)
                with open(f"test_chatterbox_{test_texts.index(test_text)}.raw", "wb") as f:
                    f.write(combined_audio)
                logger.info(f"Audio saved to test_chatterbox_{test_texts.index(test_text)}.raw")
            
            # Small delay between tests
            await asyncio.sleep(1.0)
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        await service.stop(start_frame)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_chatterbox_websocket_tts())