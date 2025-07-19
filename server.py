#!/usr/bin/env python3
"""
Optimized Chatterbox TTS WebSocket Server

FIXED: Ensures model stays loaded in memory between requests
FIXED: Proper model persistence and memory management
"""
import numpy as np
import asyncio
import json
import base64
import logging
import argparse
from typing import Optional, Dict, Any, AsyncGenerator, Tuple
from pathlib import Path
import threading
import time
from aiohttp import WSMsgType

import websockets
from websockets.server import WebSocketServerProtocol
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
CONNECTION_LIMIT = 5
connection_semaphore = asyncio.Semaphore(CONNECTION_LIMIT)
async def safe_send(ws: WebSocketServerProtocol, payload: str):
    try:
        if hasattr(ws, "closed") and ws.closed:
            return
        if isinstance(payload, (str, bytes)):
            await ws.send(payload)
        else:
            await ws.send(json.dumps(payload))
    except (ConnectionClosedOK, ConnectionClosedError, AttributeError) as e:
        logger.warning(f"safe_send: websocket closed or error: {e}")
        return
    except Exception as e:
        logger.warning(f"safe_send: other send error: {e}")
        return

        
class OptimizedChatterboxServer:
    def __init__(
        self,
        model: ChatterboxTTS,
        voice_prompt_path: Optional[str] = None,
        default_params: Optional[Dict[str, Any]] = None
    ):
        self.model = model
        self.voice_prompt_path = voice_prompt_path
        self.default_params = default_params or {
            "chunk_size": 25,
            "exaggeration": 0.5,
            "temperature": 0.8,
            "cfg_weight": 0.5,
            "context_window": 50,
            "fade_duration": 0.05
        }
        
        # Model state tracking
        self.model_loaded = False
        self.model_lock = threading.Lock()
        self.total_requests = 0
        
        # Pre-warm the model with voice prompt if provided
        if voice_prompt_path:
            logger.info(f"🎭 Loading voice prompt from: {voice_prompt_path}")
            self._prepare_model()
    
    def _prepare_model(self):
        """Ensure model is properly loaded and ready"""
        with self.model_lock:
            if not self.model_loaded:
                logger.info("Preparing model for first use...")
                
                # Ensure model is in eval mode
                # self.model.eval()
                
                # Pre-load voice conditionals if needed
                if self.voice_prompt_path:
                    logger.info("Pre-loading voice conditionals...")
                    self.model.prepare_conditionals(
                        self.voice_prompt_path, 
                        exaggeration=self.default_params["exaggeration"]
                    )
                
                # Run a tiny test synthesis to ensure everything is loaded
                logger.info("🧪 Running test synthesis to warm up model...")
                test_start = time.time()
                try:
                    with torch.no_grad():
                        _ = self.model.generate(
                            "Test",
                            exaggeration=0.1,
                            cfg_weight=0.1,
                            temperature=0.1
                        )
                    test_time = time.time() - test_start
                    logger.info(f"Model warmed up successfully in {test_time:.2f}s")
                    self.model_loaded = True
                except Exception as e:
                    logger.error(f"Model warmup failed: {e}")
                    raise

    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle a WebSocket connection with a single recv loop."""
        async with connection_semaphore:
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
            self.total_requests += 1
            logger.info(f"🔗 New connection #{self.total_requests} from {client_id}")

            # Ensure model is ready before processing any requests
            if not self.model_loaded:
                self._prepare_model()

            # Unified receive loop
            while True:
                try:
                    #only one active loop for the web socket receive
                    raw = await websocket.recv() 
                except (ConnectionClosedOK, ConnectionClosedError):
                    logger.info(f"🔌 Client {client_id} disconnected.")
                    break

                # Parse JSON
                try:
                    data = json.loads(raw)
                except Exception:
                    await safe_send(websocket, {
                        "type": "error",
                        "error": "Invalid JSON"
                    })
                    continue

                # Dispatch based on message type
                msg_type = data.get("type")
                if msg_type == "synthesize":
                    await self.handle_synthesize(websocket, data)
                elif msg_type == "stream_start":
                    await self.handle_stream(websocket, data)
                    logger.info("Returned from handle_stream, waiting for next message!")
                elif msg_type == "ping":
                    await safe_send(websocket, {"type": "pong"})
                else:
                    await safe_send(websocket, json.dumps({
                        "type": "error",
                        "error": f"Unknown message type: {msg_type}"
                    }))

            logger.info(f"🧹 Cleaned up connection from {client_id}")
    
    async def process_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Process incoming WebSocket messages"""
        msg_type = data.get("type")
        
        if msg_type == "synthesize":
            await self.handle_synthesize(websocket, data)
        elif msg_type == "stream_start":
            await self.handle_stream(websocket, data)
        else:
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Unknown message type: {msg_type}"
            }))
    def convert_audio_to_pcm16(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert float32 audio tensor to PCM16 bytes for LiveKit compatibility"""
        # Ensure audio is on CPU and convert to numpy
        audio_np = audio_tensor.squeeze(0).cpu().numpy()
        
        # Clip to prevent overflow
        audio_np = np.clip(audio_np, -1.0, 1.0)
        
        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Return as bytes
        
        return audio_int16.tobytes()
    async def handle_synthesize(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle non-streaming synthesis request"""
        text = data.get("text", "")
        request_id = data.get("request_id", "unknown")
        params = data.get("params", {})
        
        if not text:
            await websocket.send(json.dumps({
                "type": "error",
                "error": "No text provided",
                "request_id": request_id
            }))
            return
        
        # Merge with default params
        synthesis_params = {**self.default_params, **params}
        
        logger.info(f"Synthesizing (req: {request_id}): {text[:50]}...")
        logger.info(f"Model already loaded: {self.model_loaded}")
        
        synthesis_start = time.time()
        
        try:
            # Use the already-loaded model - NO RELOADING!
            logger.info("⚡ Using pre-loaded model (no sampling bar should appear)")
            
            wav = await asyncio.to_thread(
                self._generate_with_loaded_model,
                text,
                synthesis_params
            )
            
            synthesis_time = time.time() - synthesis_start
            logger.info(f"Pure synthesis time: {synthesis_time:.3f}s")
            
            # Convert to bytes
            audio_bytes = await asyncio.to_thread(
                # lambda: wav.squeeze(0).cpu().numpy().astype('int16').tobytes()
                lambda: self.convert_audio_to_pcm16(wav)
            )
            
            # Send response
            await websocket.send(json.dumps({
                "type": "audio",
                "request_id": request_id,
                "audio_content": base64.b64encode(audio_bytes).decode('utf-8'),
                "sample_rate": self.model.sr,
                "channels": 1,
                "format": "pcm_16"
            }))
            
            # Send completion
            await websocket.send(json.dumps({
                "type": "synthesis_complete",
                "request_id": request_id
            }))
            
            logger.info(f"Synthesis completed in {synthesis_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": str(e),
                "request_id": request_id
            }))
    
    def _generate_with_loaded_model(self, text: str, params: Dict[str, Any]) -> torch.Tensor:
        """Generate audio using the already-loaded model"""
        # Use no_grad to prevent gradient computation and ensure faster inference
        with torch.no_grad():
            return self.model.generate(
                text,
                audio_prompt_path=self.voice_prompt_path if not self.model.conds else None,
                exaggeration=params["exaggeration"],
                cfg_weight=params["cfg_weight"],
                temperature=params["temperature"]
            )
    
    #     # After process_stream exits, control returns to main loop
    async def handle_stream(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle streaming synthesis request with incremental processing"""
        session_id = data.get("session_id", "unknown")
        params = data.get("params", {})
        
        # Merge with default params
        synthesis_params = {**self.default_params, **params}
        
        logger.info(f"🎵 Starting stream session: {session_id}")
        logger.info(f"📊 Model loaded: {self.model_loaded}")
        
        # Send acknowledgment
        await safe_send(websocket, {
            "type": "stream_ready",
            "session_id": session_id,
            "sample_rate": self.model.sr,
            "channels": 1,
            "format": "pcm_16"
        })
        
        # Buffer for accumulating text
        text_buffer = ""
        stream_active = True
        chunk_index = 0  # Track chunk index across incremental calls
        
        while stream_active:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                data = json.loads(message)
                if data.get("type") == "stream_text":
                    text_buffer += data.get("text", "")
                    # Process text incrementally if buffer exceeds 20 characters
                    if len(text_buffer) >= 10:
                        async for audio_chunk, metrics in self.async_generate_stream(text_buffer, synthesis_params):
                            header = json.dumps({
                                "type": "audio_chunk",
                                "session_id": session_id,
                                "chunk_index": chunk_index,
                                "sample_rate": self.model.sr,
                                "channels": 1,
                                "format": "pcm_16"
                            })
                            try:
                                await safe_send(websocket, header)
                                audio_bytes = self.convert_audio_to_pcm16(audio_chunk)
                                await websocket.send(audio_bytes)
                                logger.info(f"Sent chunk {chunk_index}, len={len(audio_chunk)}")
                                chunk_index += 1
                            except (ConnectionClosedOK, ConnectionClosedError, AttributeError) as e:
                                logger.warning(f" WebSocket closed or error during sending chunk: {e}")
                                stream_active = False
                                break
                        # Keep text_buffer for context in next iteration
                        text_buffer = ""
                elif data.get("type") == "stream_flush":
                    if text_buffer:
                        async for audio_chunk, metrics in self.async_generate_stream(text_buffer, synthesis_params):
                            header = json.dumps({
                                "type": "audio_chunk",
                                "session_id": session_id,
                                "chunk_index": chunk_index,
                                "sample_rate": self.model.sr,
                                "channels": 1,
                                "format": "pcm_16"
                            })
                            try:
                                await safe_send(websocket, header)
                                audio_bytes = self.convert_audio_to_pcm16(audio_chunk)
                                await websocket.send(audio_bytes)
                                logger.info(f"Sent chunk {chunk_index}, len={len(audio_chunk)}")
                                chunk_index += 1
                            except (ConnectionClosedOK, ConnectionClosedError, AttributeError) as e:
                                logger.warning(f"🔌 WebSocket closed or error during sending chunk: {e}")
                                stream_active = False
                                break
                        text_buffer = ""
                elif data.get("type") == "stream_end":
                    if text_buffer:
                        async for audio_chunk, metrics in self.async_generate_stream(text_buffer, synthesis_params):
                            header = json.dumps({
                                "type": "audio_chunk",
                                "session_id": session_id,
                                "chunk_index": chunk_index,
                                "sample_rate": self.model.sr,
                                "channels": 1,
                                "format": "pcm_16"
                            })
                            try:
                                await safe_send(websocket, header)
                                audio_bytes = self.convert_audio_to_pcm16(audio_chunk)
                                await websocket.send(audio_bytes)
                                logger.info(f"Sent chunk {chunk_index}, len={len(audio_chunk)}")
                                chunk_index += 1
                            except (ConnectionClosedOK, ConnectionClosedError, AttributeError) as e:
                                logger.warning(f"🔌 WebSocket closed or error during sending chunk: {e}")
                                break
                        text_buffer = ""
                    await safe_send(websocket, {
                        "type": "stream_complete", 
                        "session_id": session_id
                    })
                    stream_active = False
            except asyncio.TimeoutError:
                logger.warning(f"Stream timeout for session: {session_id}")
                break
            except (ConnectionClosedOK, ConnectionClosedError):
                logger.info(f"Client closed stream session: {session_id}")
                break
            except Exception as e:
                logger.error(f"Stream error: {e}")
                break
    
    async def stream_synthesis(
        self, 
        websocket: WebSocketServerProtocol, 
        text: str, 
        session_id: str,
        params: Dict[str, Any]
    ):
        """Stream synthesis for given text"""
        logger.info(f"🎵 Streaming synthesis for session {session_id}: {text[:50]}...")
        logger.info(f"⚡ Using pre-loaded model for streaming")
        
        stream_start = time.time()
        logger.info(f"websocket type in stream_synthesis: {type(websocket)}")
        chunk_index = 0
        try:

            
            # Run streaming synthesis with pre-loaded model
            async for audio_chunk, metrics in self.async_generate_stream(text, params):
                # INSTEAD OF checking websocket.closed, just try sending and catch exceptions

                # Send audio chunk header
                header = json.dumps({
                    "type": "audio_chunk",
                    "session_id": session_id,
                    "chunk_index": chunk_index,
                    "sample_rate": self.model.sr,
                    "channels": 1,
                    "format": "pcm_16"
                })

                try:
                    await safe_send(websocket, header)
                    audio_bytes = self.convert_audio_to_pcm16(audio_chunk)
                    # await safe_send(websocket, self.convert_audio_to_pcm16(audio_chunk))  # This sends BINARY frame
                    await websocket.send(audio_bytes)
                    logger.info(f"Sent chunk {chunk_index}, len={len(audio_chunk)}")
                    chunk_index += 1
                except (ConnectionClosedOK, ConnectionClosedError, AttributeError) as e:
                    logger.warning(f"🔌 WebSocket closed or error during sending chunk: {e}")
                    break  # Exit the chunk loop on any send error
                    
                except Exception as e:
                    logger.warning(f"Other send error (during streaming chunk): {e}")
                    break
            
            stream_time = time.time() - stream_start
            logger.info(f"Stream synthesis completed in {stream_time:.3f}s")
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.info(f"🔌 WebSocket closed during stream synthesis for session {session_id}")
        except Exception as e:
            logger.error(f"Stream synthesis error: {e}")
    
    async def async_generate_stream(self, text: str, params: Dict[str, Any]) -> AsyncGenerator[Tuple[torch.Tensor, Any], None]:
        """
        Async wrapper for generate_stream with proper StopIteration handling
        
        FIXED: Uses torch.no_grad() instead of inference_mode and properly handles tensor copying
        """
        loop = asyncio.get_event_loop()
        
        def _create_generator():
            """Create the generator in a thread-safe way"""
            try:
                # Wrap the entire generator creation in no_grad context
                with torch.no_grad():
                    return self.model.generate_stream(
                        text,
                        audio_prompt_path=self.voice_prompt_path if not self.model.conds else None,
                        exaggeration=params["exaggeration"],
                        cfg_weight=params["cfg_weight"],
                        temperature=params["temperature"],
                        chunk_size=params["chunk_size"],
                        context_window=params["context_window"],
                        fade_duration=params["fade_duration"],
                        print_metrics=False
                    )
            except Exception as e:
                logger.error(f"Error creating generator: {e}")
                return None
        
        # Create generator in executor
        generator = await loop.run_in_executor(None, _create_generator)
        
        if generator is None:
            return
        
        # Yield chunks with proper exception handling
        while True:
            try:
                # Get next chunk in executor to handle StopIteration properly
                def _get_next():
                    try:
                        with torch.no_grad():  # Ensure no_grad context for chunk processing
                            chunk_data = next(generator)
                            if isinstance(chunk_data, tuple) and len(chunk_data) == 2:
                                audio_chunk, metrics = chunk_data
                                # Create safe copy of the audio chunk
                                safe_chunk = audio_chunk.detach().clone()
                                return (safe_chunk, metrics)
                            return chunk_data
                    except StopIteration:
                        return None  # Convert StopIteration to None
                    except Exception as e:
                        logger.error(f"Error getting next chunk: {e}")
                        return None
                
                chunk = await loop.run_in_executor(None, _get_next)
                
                if chunk is None:
                    # Generator exhausted or error occurred
                    break
                    
                yield chunk
                
            except Exception as e:
                logger.error(f"Error in async generator: {e}")
                break

async def main():
    parser = argparse.ArgumentParser(description="Optimized Chatterbox TTS WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--device", default="auto", help="Device to use (cuda/mps/cpu/auto)")
    parser.add_argument("--model-path", type=str, help="Path to local model checkpoint")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    
    logger.info(f"🖥️  Using device: {device}")
    
    # Load model ONCE at startup
    logger.info("Loading Chatterbox model at startup...")
    start_time = time.time()
    
    # Load model with torch.no_grad() context
    logger.info("Loading Chatterbox model...")
    with torch.no_grad():
        if args.model_path:
            model = ChatterboxTTS.from_local(args.model_path, device)
        else:
            model = ChatterboxTTS.from_pretrained(device=device)
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded successfully in {load_time:.2f}s")
    
    # Create server with pre-loaded model
    server = OptimizedChatterboxServer(
        model=model,
        voice_prompt_path="/home/user/voice/styletts/audio/emotions/base/wavs/Base-1.wav",
        default_params={
            "chunk_size": 75,
            "exaggeration": 0.5,
            "temperature": 0.8,
            "cfg_weight": 0.5,
            "context_window": 70,
            "fade_duration": 0.09
        }
    )
    
    # Start WebSocket server
    logger.info(f"Starting WebSocket server on {args.host}:{args.port}")
    logger.info(f"Model is pre-loaded and ready for requests!")
    
    async with websockets.serve(server.handle_connection, args.host, args.port, max_size=10*1024*1024):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())