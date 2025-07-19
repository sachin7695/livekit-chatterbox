"""
LiveKit Plugin for Chatterbox TTS

This plugin provides integration with a Chatterbox TTS WebSocket server.
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, replace
from typing import Optional, Dict, Any
import weakref
from aiohttp import WSMsgType

import time
import aiohttp
import numpy as np

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.metrics import TTSMetrics
from collections.abc import AsyncIterable
from livekit.agents.tts import TTSError
from livekit.agents.utils import aio
import logging
logger = logging.getLogger("livekit.plugins.chatterbox")

DEFAULT_WEBSOCKET_URL = "ws://localhost:8765"
BUFFERED_WORDS_COUNT = 5


@dataclass
class _TTSOptions:
    chunk_size: int
    exaggeration: float
    temperature: float
    cfg_weight: float
    context_window: int
    fade_duration: float
    sample_rate: int
    tokenizer: tokenize.SentenceTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        websocket_url: str = DEFAULT_WEBSOCKET_URL,
        chunk_size: int = 25,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        context_window: int = 50,
        fade_duration: float = 0.05,
        sample_rate: int = 24000,  # Chatterbox default
        tokenizer: tokenize.SentenceTokenizer | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of the Chatterbox TTS plugin.

        Args:
            websocket_url: URL of the Chatterbox WebSocket server
            chunk_size: Number of tokens per chunk for streaming
            exaggeration: Emotion exaggeration factor
            temperature: Sampling temperature
            cfg_weight: Classifier-free guidance weight
            context_window: Context window size for streaming
            fade_duration: Fade duration in seconds
            sample_rate: Audio sample rate in Hz
            tokenizer: Sentence tokenizer for streaming
            http_session: Optional aiohttp session
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,  # Chatterbox outputs mono audio
        )
        
        self._websocket_url = websocket_url
        
        if tokenizer is None:
            tokenizer = tokenize.basic.SentenceTokenizer(
                min_sentence_len=BUFFERED_WORDS_COUNT
            )
        
        self._opts = _TTSOptions(
            chunk_size=chunk_size,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            context_window=context_window,
            fade_duration=fade_duration,
            sample_rate=sample_rate,
            tokenizer=tokenizer,
        )
        
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )
    
    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        """Connect to the Chatterbox WebSocket server"""
        logger.debug(f"Connecting to Chatterbox WebSocket: {self._websocket_url}")
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(self._websocket_url),
            timeout,
        )
    
    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Close WebSocket connection"""
        await ws.close()
    
    def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an aiohttp session"""
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session
    
    def prewarm(self) -> None:
        """Pre-warm the connection pool"""
        self._pool.prewarm()
    
    def update_options(
        self,
        *,
        chunk_size: int | None = None,
        exaggeration: float | None = None,
        temperature: float | None = None,
        cfg_weight: float | None = None,
    ) -> None:
        """Update TTS configuration options"""
        if chunk_size is not None:
            self._opts.chunk_size = chunk_size
        if exaggeration is not None:
            self._opts.exaggeration = exaggeration
        if temperature is not None:
            self._opts.temperature = temperature
        if cfg_weight is not None:
            self._opts.cfg_weight = cfg_weight
    
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize text to speech (non-streaming)"""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
    
    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming synthesis session"""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream
    
    async def aclose(self) -> None:
        """Close all resources"""
        for stream in list(self._streams):
            await stream.aclose()
        
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming synthesis using Chatterbox"""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                request_id = utils.shortuuid()

                # Send synthesis request
                await ws.send_str(json.dumps({
                    "type": "synthesize",
                    "text": self._input_text,
                    "request_id": request_id,
                    "params": {
                        "exaggeration": self._opts.exaggeration,
                        "temperature": self._opts.temperature,
                        "cfg_weight": self._opts.cfg_weight,
                    }
                }))

                # Initialize emitter
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                while True:
                    msg = await ws.receive()

                    # Handle clean close
                    if msg.type in (WSMsgType.CLOSED, WSMsgType.CLOSE, WSMsgType.CLOSING):
                        raise APIConnectionError("WebSocket connection closed unexpectedly")

                    # TEXT = control frames (errors / done)
                    if msg.type == WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "synthesis_complete":
                            break
                        if data.get("type") == "error":
                            raise APIError(f"Server error: {data['error']}")
                        # otherwise ignore
                        continue

                    # BINARY = raw PCM16 audio
                    if msg.type == WSMsgType.BINARY:
                        pcm_bytes = msg.data
                        output_emitter.push(pcm_bytes)
                        continue

                    # ignore PING/PONG/etc.
                    continue

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError(f"Connection error: {e}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis using Chatterbox"""
    
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()
        # Add metrics monitoring like Resemble AI does
        self._tee = aio.itertools.tee(self._event_ch, 2)
        self._event_aiter, monitor_aiter = self._tee
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="TTS._metrics_task"
        )
    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[tts.SynthesizedAudio]) -> None:
        """Task used to collect metrics"""
        start_time = time.perf_counter()
        audio_duration = 0.0
        ttfb = -1.0
        request_id = ""
        segment_id = ""

        def _emit_metrics() -> None:
            nonlocal audio_duration, ttfb, request_id, segment_id
            
            # FIXED: Better check for started_time
            if not hasattr(self, '_started_time') or self._started_time == 0:
                return
                
            duration = time.perf_counter() - self._started_time
            
            # FIXED: Better text length handling
            text_len = 0
            if hasattr(self, '_mtc_text') and self._mtc_text:
                text_len = len(self._mtc_text)
            elif hasattr(self, '_pushed_text') and self._pushed_text:
                text_len = len(self._pushed_text)
            
            metrics = TTSMetrics(
                timestamp=time.time(),
                request_id=request_id,
                segment_id=segment_id,
                ttfb=ttfb,
                duration=duration,
                characters_count=text_len,
                audio_duration=audio_duration,
                cancelled=self._task.cancelled(),
                label=self._tts._label,
                streamed=True,
            )
            self._tts.emit("metrics_collected", metrics)
            
            # Reset for next segment
            audio_duration = 0.0
            ttfb = -1.0
            self._started_time = 0

        async for ev in event_aiter:
            if ttfb == -1.0:
                ttfb = time.perf_counter() - start_time
            audio_duration += ev.frame.duration
            request_id = ev.request_id
            segment_id = ev.segment_id
            
            if ev.is_final:
                _emit_metrics()


    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run streaming synthesis"""
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )
        
        async def _tokenize_input() -> None:
            """Tokenize text input into sentences"""
            input_stream = None
            async for text in self._input_ch:
                if isinstance(text, str):
                    if input_stream is None:
                        # New segment
                        input_stream = self._opts.tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(text)
                elif isinstance(text, self._FlushSentinel):
                    if input_stream is not None:
                        input_stream.end_input()
                    input_stream = None
            
            if input_stream is not None:
                input_stream.end_input()
            
            self._segments_ch.close()
        
        async def _process_segments() -> None:
            """Process tokenized segments"""
            async for input_stream in self._segments_ch:
                await self._run_ws(input_stream, output_emitter)
        
        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            self._tts.emit("error", TTSError(
                timestamp=time.time(),
                label=self._tts._label,
                error=e,
                recoverable=True
            ))
            raise APIConnectionError(f"Connection error: {e}") from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
    
    async def _run_ws(
        self,
        input_stream: tokenize.SentenceStream,
        output_emitter: tts.AudioEmitter
    ) -> None:
        """Run WebSocket streaming for a segment with robust close handling"""
        import aiohttp  # ensure import at top if not already
        import logging

        logger = logging.getLogger("livekit.plugins.chatterbox")

        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)
        session_id = utils.shortuuid()

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            # Start streaming session
            await ws.send_str(json.dumps({
                "type": "stream_start",
                "session_id": session_id,
                "params": {
                    "chunk_size": self._opts.chunk_size,
                    "exaggeration": self._opts.exaggeration,
                    "temperature": self._opts.temperature,
                    "cfg_weight": self._opts.cfg_weight,
                    "context_window": self._opts.context_window,
                    "fade_duration": self._opts.fade_duration,
                }
            }))

            # Wait for ready
            msg = await ws.receive()
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise APIError("Expected stream_ready response")
            data = json.loads(msg.data)
            if data["type"] != "stream_ready":
                raise APIError(f"Unexpected response: {data['type']}")

            send_task_cancelled = False

            async def _send_task():
                """Send text chunks to server; stop on websocket close."""
                nonlocal send_task_cancelled
                text_buffer = ""
                try:
                    async for sentence_data in input_stream:
                        if ws.closed:
                            logger.warning("[SEND] WebSocket closed. Exiting send loop.")
                            break
                        text_buffer += sentence_data.token

                        if len(text_buffer) > 50:
                            self._mark_started()
                            if hasattr(self, '_mtc_pending_texts'):
                                self._mtc_pending_texts.append(text_buffer)
                            await ws.send_str(json.dumps({
                                "type": "stream_text",
                                "text": text_buffer
                            }))
                            text_buffer = ""

                    if ws.closed:
                        logger.warning("[SEND] WebSocket closed after loop. Not sending flush/end.")
                        return

                    # Send any remaining text
                    if text_buffer:
                        self._mark_started()
                        await ws.send_str(json.dumps({
                            "type": "stream_text",
                            "text": text_buffer
                        }))

                    # Only send flush/end if still open
                    if not ws.closed:
                        await ws.send_str(json.dumps({"type": "stream_flush"}))
                    if not ws.closed:
                        await ws.send_str(json.dumps({"type": "stream_end"}))

                except (aiohttp.ClientConnectionError, aiohttp.ClientConnectionResetError) as e:
                    logger.warning(f"[SEND] Connection error: {e}")
                except Exception as e:
                    logger.warning(f"[SEND] Unexpected error: {e}")
                finally:
                    send_task_cancelled = True

            async def _recv_task():
                """Receive audio and control messages from server. Cancels send if stream is done."""
                try:
                    while True:
                        msg = await ws.receive()

                        # Handle clean close
                        if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING):
                            logger.info("[RECV] WebSocket closed by server.")
                            break

                        # TEXT = control frames
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if data.get("type") == "stream_complete":
                                logger.info("[RECV] Received stream_complete. Ending segment.")
                                output_emitter.end_segment()
                                # await ws.close()   #this will close the connection as soon the stream completes
                                break
                            if data.get("type") == "error":
                                logger.error(f"[RECV] Server error: {data['error']}")
                                await ws.close()
                                break
                            continue

                        # BINARY = raw PCM16 audio
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            pcm_bytes = msg.data
                            logger.info(f"[RECV] Received binary audio chunk, len={len(pcm_bytes)}")
                            output_emitter.push(pcm_bytes)
                            continue

                        # ignore PING/PONG/etc.
                except Exception as e:
                    logger.warning(f"[RECV] Unexpected error: {e}")

                finally:
                    # If this finishes first, we signal the send task should stop too
                    # if not ws.closed:
                    #     await ws.close()
                    pass

            # Run send and receive concurrently
            tasks = [
                asyncio.create_task(_send_task(), name="send_task"),
                asyncio.create_task(_recv_task(), name="recv_task"),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)

    async def aclose(self) -> None:
        """Close the stream immediately"""
        await utils.aio.cancel_and_wait(self._task)
        self._event_ch.close()
        self._input_ch.close()

        # Clean up metrics task
        if self._metrics_task is not None:
            await self._metrics_task

        await self._tee.aclose() 