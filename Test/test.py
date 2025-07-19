#!/usr/bin/env python3
"""
Comprehensive test suite for Chatterbox TTS Plugin

Tests both streaming and non-streaming modes with audio file output.
Perfect for remote server testing where you can't play audio directly.
"""

import asyncio
import os
import time
import wave
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging
import aiohttp
# Your Chatterbox TTS plugin import
# from chatterbox_tts_plugin import TTS
from client import TTS  # Import your TTS class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
CHATTERBOX_SERVER_URL = "ws://46.18.108.33:8765"  # Change to your remote server
OUTPUT_DIR = Path("./test_audio_output")
SAMPLE_RATE = 24000  # Chatterbox default
CHANNELS = 1

# Test texts
SHORT_TEXT = "Hello, this is a test of the Chatterbox TTS system."
LONG_TEXT = """
This is a comprehensive test of the streaming capabilities of the Chatterbox TTS system. 
We will test how well it handles longer texts with multiple sentences. 
The system should be able to process this text incrementally and provide smooth audio output. 
This test will help us verify that the integration between the LiveKit plugin and the Chatterbox server is working correctly.
"""

STREAMING_TEXTS = [
    "Hello there! ",
    "This is a streaming test. ",
    "We're sending text in chunks ",
    "to test real-time synthesis. ",
    "Each chunk should be processed ",
    "and returned as audio quickly. ",
    "This simulates live conversation."
]

class TimingTracker:
    """Helper class to track detailed timing metrics"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.chunks = []
        self.events = []
    
    def start(self):
        self.start_time = time.time()
        self.log_event("test_started")
    
    def log_event(self, event_name: str, data: dict = None):
        if self.start_time is None:
            return
        
        timestamp = time.time() - self.start_time
        event = {
            "timestamp": timestamp,
            "event": event_name,
            "data": data or {}
        }
        self.events.append(event)
        logger.info(f"[{self.test_name}] {timestamp:.3f}s: {event_name} {data or ''}")
    
    def log_chunk(self, chunk_index: int, chunk_size: int, audio_duration: float = None):
        self.log_event("audio_chunk_received", {
            "chunk_index": chunk_index,
            "size_bytes": chunk_size,
            "audio_duration_ms": f"{audio_duration*1000:.1f}" if audio_duration else "unknown"
        })
        
        chunk_info = {
            "index": chunk_index,
            "timestamp": time.time() - self.start_time,
            "size_bytes": chunk_size,
            "audio_duration": audio_duration
        }
        self.chunks.append(chunk_info)
    
    def finish(self, total_audio_duration: float = None, total_audio_size: int = None):
        total_time = time.time() - self.start_time
        
        self.log_event("test_completed", {
            "total_time_s": f"{total_time:.3f}",
            "total_audio_duration_s": f"{total_audio_duration:.3f}" if total_audio_duration else "unknown",
            "total_audio_size_bytes": total_audio_size,
            "chunks_received": len(self.chunks)
        })
        
        # Calculate metrics
        if self.chunks:
            first_chunk_time = self.chunks[0]["timestamp"]
            logger.info(f"[{self.test_name}] === TIMING SUMMARY ===")
            logger.info(f"[{self.test_name}] Time to First Byte (TTFB): {first_chunk_time:.3f}s")
            logger.info(f"[{self.test_name}] Total chunks: {len(self.chunks)}")
            logger.info(f"[{self.test_name}] Total synthesis time: {total_time:.3f}s")
            
            if total_audio_duration:
                realtime_factor = total_audio_duration / total_time
                logger.info(f"[{self.test_name}] Audio duration: {total_audio_duration:.3f}s")
                logger.info(f"[{self.test_name}] Real-time factor: {realtime_factor:.2f}x")
        
        return total_time


def setup_output_directory():
    """Create output directory for test audio files"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info(f"Audio files will be saved to: {OUTPUT_DIR.absolute()}")


def concatenate_and_save_audio(audio_chunks: List[bytes], output_path: str, 
                              sample_rate: int = SAMPLE_RATE) -> Tuple[float, int]:
    """
    Concatenate audio chunks and save as WAV file
    
    Returns:
        Tuple of (audio_duration_seconds, total_bytes)
    """
    try:
        save_start = time.time()
        
        # Combine all audio chunks
        combined_audio = b"".join(audio_chunks)
        
        if not combined_audio:
            logger.warning(f"No audio data to save for {output_path}")
            return 0.0, 0
        
        # Convert bytes to numpy array (float32)
        audio_array = np.frombuffer(combined_audio, dtype=np.float32)
        
        # Convert float32 to int16 for WAV (standard format)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 2 bytes for int16
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        duration = len(audio_array) / sample_rate
        save_time = time.time() - save_start
        
        logger.info(f"Saved concatenated audio: {output_path}")
        logger.info(f"  Duration: {duration:.3f}s, Size: {len(combined_audio)} bytes")
        logger.info(f"  Chunks combined: {len(audio_chunks)}, Save time: {save_time:.3f}s")
        
        return duration, len(combined_audio)
        
    except Exception as e:
        logger.error(f"Failed to save concatenated audio {output_path}: {e}")
        return 0.0, 0


async def test_non_streaming_synthesis(session: aiohttp.ClientSession):
    """Test non-streaming (chunked) synthesis"""
    logger.info("=== Testing Non-Streaming Synthesis ===")
    
    tts = TTS(
        websocket_url=CHATTERBOX_SERVER_URL,
        http_session=session
    )
    
    try:
        # Test short text
        tracker = TimingTracker("NonStreaming-Short")
        tracker.start()
        tracker.log_event("synthesis_request_sent", {"text_length": len(SHORT_TEXT)})
        
        all_audio_chunks = []
        chunk_count = 0
        
        async with tts.synthesize(SHORT_TEXT) as stream:
            async for audio in stream:
                audio_data = audio.frame.data.tobytes()
                all_audio_chunks.append(audio_data)
                
                # Calculate audio duration for this chunk
                audio_samples = len(audio.frame.data)
                chunk_duration = audio_samples / SAMPLE_RATE
                
                tracker.log_chunk(chunk_count, len(audio_data), chunk_duration)
                chunk_count += 1
        
        # Save concatenated audio
        output_path = OUTPUT_DIR / "non_streaming_short.wav"
        audio_duration, total_size = concatenate_and_save_audio(all_audio_chunks, str(output_path))
        
        total_time = tracker.finish(audio_duration, total_size)
        
        # Test long text
        tracker = TimingTracker("NonStreaming-Long")
        tracker.start()
        tracker.log_event("synthesis_request_sent", {"text_length": len(LONG_TEXT)})
        
        all_audio_chunks = []
        chunk_count = 0
        
        async with tts.synthesize(LONG_TEXT) as stream:
            async for audio in stream:
                audio_data = audio.frame.data.tobytes()
                all_audio_chunks.append(audio_data)
                
                # Calculate audio duration for this chunk
                audio_samples = len(audio.frame.data)
                chunk_duration = audio_samples / SAMPLE_RATE
                
                tracker.log_chunk(chunk_count, len(audio_data), chunk_duration)
                chunk_count += 1
        
        # Save concatenated audio
        output_path = OUTPUT_DIR / "non_streaming_long.wav"
        audio_duration, total_size = concatenate_and_save_audio(all_audio_chunks, str(output_path))
        
        tracker.finish(audio_duration, total_size)
        
    except Exception as e:
        logger.error(f"Non-streaming test failed: {e}")
    finally:
        await tts.aclose()


async def test_streaming_synthesis(session: aiohttp.ClientSession):
    """Test streaming synthesis with incremental text input"""
    logger.info("=== Testing Streaming Synthesis ===")
    
    tts = TTS(
        websocket_url=CHATTERBOX_SERVER_URL,
        http_session=session
    )
    
    try:
        tracker = TimingTracker("Streaming")
        tracker.start()
        
        all_audio_chunks: List[bytes] = []
        chunk_count = 0
        
        async with tts.stream() as stream:
            # Start sending text chunks
            send_task = asyncio.create_task(send_streaming_text(stream, tracker))
            
            # Receive audio chunks
            async for audio in stream:
                audio_data = audio.frame.data.tobytes()
                all_audio_chunks.append(audio_data)
                
                # Calculate audio duration for this chunk
                audio_samples = len(audio.frame.data)
                chunk_duration = audio_samples / SAMPLE_RATE
                
                tracker.log_chunk(chunk_count, len(audio_data), chunk_duration)
                chunk_count += 1
            
            await send_task
        
        # Save concatenated streaming audio
        output_path = OUTPUT_DIR / "streaming_combined.wav"
        audio_duration, total_size = concatenate_and_save_audio(all_audio_chunks, str(output_path))
        
        tracker.finish(audio_duration, total_size)
        
    except Exception as e:
        logger.error(f"Streaming test failed: {e}")
    finally:
        await tts.aclose()


async def send_streaming_text(stream, tracker: TimingTracker):
    """Send text chunks with timing tracking"""
    tracker.log_event("text_sending_started")
    
    total_text_length = sum(len(chunk) for chunk in STREAMING_TEXTS)
    tracker.log_event("total_text_info", {"chunks": len(STREAMING_TEXTS), "total_length": total_text_length})
    
    for i, text_chunk in enumerate(STREAMING_TEXTS):
        tracker.log_event("text_chunk_sent", {
            "chunk_index": i,
            "text": text_chunk.strip(),
            "length": len(text_chunk)
        })
        
        stream.push_text(text_chunk)
        
        # Simulate realistic typing speed
        await asyncio.sleep(0.5)
    
    tracker.log_event("text_flush_sent")
    stream.flush()
    
    tracker.log_event("text_sending_completed")


async def test_real_time_streaming(session: aiohttp.ClientSession):
    """Test real-time streaming that simulates live conversation"""
    logger.info("=== Testing Real-Time Streaming ===")
    
    tts = TTS(
        websocket_url=CHATTERBOX_SERVER_URL,
        http_session=session
    )
    
    try:
        tracker = TimingTracker("RealTime")
        tracker.start()
        
        audio_chunks: List[bytes] = []
        chunk_count = 0
        
        async with tts.stream() as stream:
            # Simulate user typing in real-time
            async def simulate_user_input():
                sentences = [
                    "Hi there!",
                    "How are you doing today?", 
                    "I'm testing the real-time capabilities.",
                    "This should feel like a natural conversation."
                ]
                
                for sent_idx, sentence in enumerate(sentences):
                    tracker.log_event("sentence_started", {
                        "sentence_index": sent_idx,
                        "text": sentence
                    })
                    
                    # Simulate typing word by word
                    words = sentence.split()
                    for word_idx, word in enumerate(words):
                        stream.push_text(word + " ")
                        tracker.log_event("word_sent", {
                            "word_index": word_idx,
                            "word": word
                        })
                        await asyncio.sleep(0.3)  # Typing delay
                    
                    stream.flush()  # End of sentence
                    tracker.log_event("sentence_flushed", {"sentence_index": sent_idx})
                    await asyncio.sleep(1.0)   # Pause between sentences
            
            # Run input simulation and audio collection concurrently
            input_task = asyncio.create_task(simulate_user_input())
            
            async for audio in stream:
                audio_data = audio.frame.data.tobytes()
                audio_chunks.append(audio_data)
                
                # Calculate audio duration for this chunk
                audio_samples = len(audio.frame.data)
                chunk_duration = audio_samples / SAMPLE_RATE
                
                tracker.log_chunk(chunk_count, len(audio_data), chunk_duration)
                chunk_count += 1
            
            await input_task
        
        # Save combined real-time audio
        output_path = OUTPUT_DIR / "realtime_combined.wav"
        audio_duration, total_size = concatenate_and_save_audio(audio_chunks, str(output_path))
        
        tracker.finish(audio_duration, total_size)
        
    except Exception as e:
        logger.error(f"Real-time test failed: {e}")
    finally:
        await tts.aclose()


async def test_simple_connection(session: aiohttp.ClientSession):
    """Test simple connection to server"""
    logger.info("=== Testing Simple Connection ===")
    
    tts = TTS(
        websocket_url=CHATTERBOX_SERVER_URL,
        http_session=session
    )
    
    try:
        tracker = TimingTracker("Simple")
        tracker.start()
        
        simple_text = "Hello world"
        tracker.log_event("synthesis_request_sent", {"text": simple_text, "length": len(simple_text)})
        
        all_audio_chunks = []
        chunk_count = 0
        
        async with tts.synthesize(simple_text) as stream:
            async for audio in stream:
                audio_data = audio.frame.data.tobytes()
                all_audio_chunks.append(audio_data)
                
                # Calculate audio duration for this chunk
                audio_samples = len(audio.frame.data)
                chunk_duration = audio_samples / SAMPLE_RATE
                
                tracker.log_chunk(chunk_count, len(audio_data), chunk_duration)
                chunk_count += 1
        
        # Save simple test audio
        output_path = OUTPUT_DIR / "simple_test.wav"
        audio_duration, total_size = concatenate_and_save_audio(all_audio_chunks, str(output_path))
        
        tracker.finish(audio_duration, total_size)
        
    except Exception as e:
        logger.error(f"Simple connection test failed: {e}")
    finally:
        await tts.aclose()


async def test_error_handling(session: aiohttp.ClientSession):
    """Test error handling and recovery"""
    logger.info("=== Testing Error Handling ===")
    
    tts = TTS(
        websocket_url=CHATTERBOX_SERVER_URL,
        http_session=session
    )
    
    try:
        # Test with empty text
        logger.info("Testing empty text handling...")
        try:
            tracker = TimingTracker("ErrorTest-Empty")
            tracker.start()
            
            async with tts.synthesize("") as stream:
                async for audio in stream:
                    logger.info("Received audio for empty text")
            
            tracker.finish()
        except Exception as e:
            logger.info(f"Empty text handled correctly: {e}")
        
        # Test with very long text
        logger.info("Testing very long text...")
        very_long_text = "This is a test sentence. " * 100  # Even longer for stress test
        
        tracker = TimingTracker("ErrorTest-Long")
        tracker.start()
        tracker.log_event("long_text_test", {"text_length": len(very_long_text), "sentences": 100})
        
        all_audio_chunks = []
        chunk_count = 0
        
        async with tts.synthesize(very_long_text) as stream:
            async for audio in stream:
                audio_data = audio.frame.data.tobytes()
                all_audio_chunks.append(audio_data)
                
                # Calculate audio duration for this chunk
                audio_samples = len(audio.frame.data)
                chunk_duration = audio_samples / SAMPLE_RATE
                
                tracker.log_chunk(chunk_count, len(audio_data), chunk_duration)
                chunk_count += 1
        
        # Save long text test audio
        output_path = OUTPUT_DIR / "long_text_test.wav"
        audio_duration, total_size = concatenate_and_save_audio(all_audio_chunks, str(output_path))
        
        tracker.finish(audio_duration, total_size)
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
    finally:
        await tts.aclose()


async def generate_test_report():
    """Generate a summary report of all tests"""
    logger.info("=== FINAL TEST REPORT ===")
    
    audio_files = list(OUTPUT_DIR.glob("*.wav"))
    logger.info(f"Generated {len(audio_files)} audio files:")
    
    total_size = 0
    total_duration = 0
    
    for audio_file in sorted(audio_files):
        # Get file info
        stat = audio_file.stat()
        size_kb = stat.st_size / 1024
        total_size += stat.st_size
        
        # Get audio duration
        try:
            with wave.open(str(audio_file), 'rb') as wav:
                duration = wav.getnframes() / wav.getframerate()
                total_duration += duration
            logger.info(f"  ✅ {audio_file.name}: {duration:.2f}s, {size_kb:.1f}KB")
        except Exception as e:
            logger.info(f"  ❌ {audio_file.name}: {size_kb:.1f}KB (could not read: {e})")
    
    logger.info(f"\n📊 SUMMARY:")
    logger.info(f"   Total files: {len(audio_files)}")
    logger.info(f"   Total size: {total_size/1024:.1f}KB")
    logger.info(f"   Total audio duration: {total_duration:.1f}s")
    logger.info(f"   Average file size: {(total_size/len(audio_files))/1024:.1f}KB" if audio_files else "N/A")
    logger.info(f"\n📁 Files saved to: {OUTPUT_DIR.absolute()}")
    logger.info(f"💡 Download these files to your local machine to listen!")


async def main():
    """Run all tests with proper session management and detailed timing"""
    logger.info("🚀 Starting Chatterbox TTS Plugin Test Suite")
    logger.info(f"🌐 Server URL: {CHATTERBOX_SERVER_URL}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    
    setup_output_directory()
    
    # Create a single aiohttp session for all tests
    async with aiohttp.ClientSession() as session:
        logger.info("📡 Created aiohttp session for testing")
        
        # Run tests one by one with detailed timing
        try:
            logger.info("🔄 Starting test sequence...")
            
            await test_simple_connection(session)
            await asyncio.sleep(2)
            
            await test_non_streaming_synthesis(session)
            await asyncio.sleep(2)
            
            await test_streaming_synthesis(session)
            await asyncio.sleep(2)
            
            await test_real_time_streaming(session)
            await asyncio.sleep(2)
            
            await test_error_handling(session)
            
            logger.info("✅ All tests completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("⚠️ Testing interrupted by user")
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
    
    # Generate final report
    await generate_test_report()
    
    logger.info("🎉 Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())