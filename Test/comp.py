#!/usr/bin/env python3
"""
Fixed Performance Comparison with proper timeout handling
Addresses the 24-second synthesis time and connection timeout issues
"""

import asyncio
import time
import wave
import numpy as np
from pathlib import Path
import logging
import aiohttp
from client import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHATTERBOX_SERVER_URL = "ws://46.18.108.33:8765"
OUTPUT_DIR = Path("./comparison_test")
SAMPLE_RATE = 24000

# Use shorter text since synthesis takes 24+ seconds
COMPARISON_TEXT = """
This is a performance test to compare streaming versus non-streaming synthesis.
The streaming approach should be faster because it can process text in chunks.
This test will help validate our performance expectations.
""".strip()

class FixedPerformanceComparison:
    def __init__(self):
        self.results = {}
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    def save_audio(self, audio_chunks: list, filename: str) -> tuple:
        """Save audio and return (duration, size)"""
        try:
            if not audio_chunks:
                logger.warning(f"No audio chunks to save for {filename}")
                return 0.0, 0
                
            combined_audio = b"".join(audio_chunks)
            if not combined_audio:
                logger.warning(f"Empty audio data for {filename}")
                return 0.0, 0
            
            # Convert to WAV
            audio_array = np.frombuffer(combined_audio, dtype=np.float32)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            
            output_path = OUTPUT_DIR / filename
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            duration = len(audio_array) / SAMPLE_RATE
            logger.info(f"💾 Saved: {filename} - {duration:.3f}s, {len(combined_audio)} bytes")
            return duration, len(combined_audio)
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            return 0.0, 0

    async def test_non_streaming_with_timeout(self, session: aiohttp.ClientSession):
        """Test non-streaming with extended timeout"""
        logger.info("🔄 Testing NON-STREAMING synthesis (with 60s timeout)...")
        
        # Create TTS with extended timeout
        from livekit.agents.types import APIConnectOptions
        long_timeout_options = APIConnectOptions(timeout=60.0, max_retry=1)
        
        tts = TTS(
            websocket_url=CHATTERBOX_SERVER_URL,
            http_session=session
        )
        
        try:
            start_time = time.time()
            first_audio_time = None
            audio_chunks = []
            
            logger.info(f"📝 Starting synthesis at {time.strftime('%H:%M:%S')}")
            logger.info(f"🕐 Expecting ~24 seconds based on server logs...")
            
            # Use longer timeout for synthesis
            async with tts.synthesize(COMPARISON_TEXT, conn_options=long_timeout_options) as stream:
                logger.info("🎯 Waiting for audio chunks...")
                
                chunk_count = 0
                async for audio in stream:
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        logger.info(f"🎵 First audio at {first_audio_time - start_time:.3f}s")
                    
                    audio_data = audio.frame.data.tobytes()
                    audio_chunks.append(audio_data)
                    chunk_count += 1
                    
                    if chunk_count % 5 == 0:  # Log every 5th chunk
                        elapsed = time.time() - start_time
                        logger.info(f"📦 Received {chunk_count} chunks at {elapsed:.1f}s")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"✅ NON-STREAMING completed in {total_time:.3f}s")
            
            # Save audio
            audio_duration, audio_size = self.save_audio(audio_chunks, "non_streaming_comparison.wav")
            
            # Calculate metrics
            ttfb = first_audio_time - start_time if first_audio_time else None
            
            self.results['non_streaming'] = {
                'total_time': total_time,
                'ttfb': ttfb,
                'audio_duration': audio_duration,
                'audio_size': audio_size,
                'chunks_received': len(audio_chunks),
                'realtime_factor': audio_duration / total_time if total_time > 0 else 0
            }
            
            logger.info(f"📊 NON-STREAMING RESULTS:")
            logger.info(f"   Total time: {total_time:.3f}s")
            logger.info(f"   TTFB: {ttfb:.3f}s" if ttfb else "   No TTFB measured")
            logger.info(f"   Audio duration: {audio_duration:.3f}s")
            logger.info(f"   Real-time factor: {self.results['non_streaming']['realtime_factor']:.2f}x")
            logger.info(f"   Chunks received: {len(audio_chunks)}")
            
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Non-streaming synthesis timed out after 60s")
            return False
        except Exception as e:
            logger.error(f"❌ Non-streaming test failed: {e}")
            return False
        finally:
            try:
                await tts.aclose()
            except:
                pass

    async def test_streaming_with_timeout(self, session: aiohttp.ClientSession):
        """Test streaming with extended timeout"""
        logger.info("🔄 Testing STREAMING synthesis (with 60s timeout)...")
        
        # Create TTS with extended timeout
        from livekit.agents.types import APIConnectOptions
        long_timeout_options = APIConnectOptions(timeout=60.0, max_retry=1)
        
        tts = TTS(
            websocket_url=CHATTERBOX_SERVER_URL,
            http_session=session
        )
        
        try:
            start_time = time.time()
            first_audio_time = None
            text_send_complete_time = None
            audio_chunks = []
            
            logger.info(f"📝 Starting streaming at {time.strftime('%H:%M:%S')}")
            
            async with tts.stream(conn_options=long_timeout_options) as stream:
                logger.info("📤 Sending text in chunks...")
                
                # Send text in chunks
                async def send_text():
                    nonlocal text_send_complete_time
                    
                    # Split into words for more granular streaming
                    words = COMPARISON_TEXT.split()
                    chunk_size = 5  # 5 words per chunk
                    
                    for i in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[i:i+chunk_size]) + ' '
                        logger.info(f"📤 Sending chunk {i//chunk_size + 1}: '{chunk[:30]}...'")
                        stream.push_text(chunk)
                        await asyncio.sleep(0.5)  # Simulate typing
                    
                    logger.info("🔄 Flushing stream...")
                    stream.flush()
                    text_send_complete_time = time.time()
                    logger.info(f"✅ Text sending completed at {text_send_complete_time - start_time:.3f}s")
                
                # Start sending text
                send_task = asyncio.create_task(send_text())
                
                # Receive audio chunks
                logger.info("🎯 Waiting for audio chunks...")
                chunk_count = 0
                async for audio in stream:
                    if first_audio_time is None:
                        first_audio_time = time.time()
                        logger.info(f"🎵 First audio at {first_audio_time - start_time:.3f}s")
                    
                    audio_data = audio.frame.data.tobytes()
                    audio_chunks.append(audio_data)
                    chunk_count += 1
                    
                    if chunk_count % 3 == 0:  # Log every 3rd chunk for streaming
                        elapsed = time.time() - start_time
                        logger.info(f"📦 Received {chunk_count} chunks at {elapsed:.1f}s")
                    
                    # Also log if we're taking too long
                    elapsed = time.time() - start_time
                    if elapsed > 20 and chunk_count % 10 == 0:
                        logger.info(f"⏳ Still receiving chunks... {chunk_count} received at {elapsed:.1f}s")
                
                await send_task
            
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"✅ STREAMING completed in {total_time:.3f}s")
            
            # Save audio
            audio_duration, audio_size = self.save_audio(audio_chunks, "streaming_comparison.wav")
            
            # Calculate metrics
            ttfb = first_audio_time - start_time if first_audio_time else None
            text_send_time = text_send_complete_time - start_time if text_send_complete_time else None
            
            self.results['streaming'] = {
                'total_time': total_time,
                'ttfb': ttfb,
                'text_send_time': text_send_time,
                'audio_duration': audio_duration,
                'audio_size': audio_size,
                'chunks_received': len(audio_chunks),
                'realtime_factor': audio_duration / total_time if total_time > 0 else 0
            }
            
            logger.info(f"📊 STREAMING RESULTS:")
            logger.info(f"   Total time: {total_time:.3f}s")
            logger.info(f"   TTFB: {ttfb:.3f}s" if ttfb else "   No TTFB measured")
            logger.info(f"   Text send time: {text_send_time:.3f}s" if text_send_time else "   No text send time")
            logger.info(f"   Audio duration: {audio_duration:.3f}s")
            logger.info(f"   Real-time factor: {self.results['streaming']['realtime_factor']:.2f}x")
            logger.info(f"   Chunks received: {len(audio_chunks)}")
            
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"❌ Streaming synthesis timed out after 60s")
            return False
        except Exception as e:
            logger.error(f"❌ Streaming test failed: {e}")
            return False
        finally:
            try:
                await tts.aclose()
            except:
                pass

    def analyze_results(self):
        """Compare and analyze results"""
        logger.info("\n" + "="*60)
        logger.info("📊 PERFORMANCE COMPARISON RESULTS")
        logger.info("="*60)
        
        if 'non_streaming' not in self.results or 'streaming' not in self.results:
            logger.error("❌ Incomplete results - cannot perform comparison")
            if 'non_streaming' in self.results:
                logger.info("✅ Non-streaming completed successfully")
            if 'streaming' in self.results:
                logger.info("✅ Streaming completed successfully")
            return
        
        ns = self.results['non_streaming']
        s = self.results['streaming']
        
        # Total time comparison
        time_diff = ns['total_time'] - s['total_time']
        time_improvement = (time_diff / ns['total_time']) * 100 if ns['total_time'] > 0 else 0
        
        logger.info(f"\n🏁 TOTAL SYNTHESIS TIME COMPARISON:")
        logger.info(f"   Non-streaming: {ns['total_time']:.3f}s")
        logger.info(f"   Streaming:     {s['total_time']:.3f}s")
        logger.info(f"   Difference:    {time_diff:+.3f}s")
        logger.info(f"   Improvement:   {time_improvement:+.1f}%")
        
        if time_diff > 0.5:  # Significant improvement
            logger.info(f"   🎉 STREAMING IS SIGNIFICANTLY FASTER!")
        elif time_diff > 0:
            logger.info(f"   ✅ Streaming is faster (small improvement)")
        elif time_diff < -0.5:
            logger.info(f"   ⚠️  Non-streaming is significantly faster")
        else:
            logger.info(f"   🤝 Performance is essentially equivalent")
        
        # TTFB comparison
        if ns.get('ttfb') and s.get('ttfb'):
            ttfb_diff = ns['ttfb'] - s['ttfb']
            ttfb_improvement = (ttfb_diff / ns['ttfb']) * 100 if ns['ttfb'] > 0 else 0
            
            logger.info(f"\n⚡ TIME TO FIRST AUDIO (TTFB):")
            logger.info(f"   Non-streaming: {ns['ttfb']:.3f}s")
            logger.info(f"   Streaming:     {s['ttfb']:.3f}s")
            logger.info(f"   Difference:    {ttfb_diff:+.3f}s ({ttfb_improvement:+.1f}%)")
            
            if ttfb_diff > 0:
                logger.info(f"   🚀 Streaming delivers audio {ttfb_diff:.3f}s faster!")
        
        # Audio quality verification
        duration_diff = abs(ns['audio_duration'] - s['audio_duration'])
        logger.info(f"\n🎵 AUDIO QUALITY VERIFICATION:")
        logger.info(f"   Non-streaming audio: {ns['audio_duration']:.3f}s")
        logger.info(f"   Streaming audio:     {s['audio_duration']:.3f}s")
        logger.info(f"   Duration difference: {duration_diff:.3f}s")
        
        if duration_diff < 0.1:
            logger.info(f"   ✅ Audio durations match - quality consistent!")
        else:
            logger.info(f"   ⚠️  Significant duration difference - check audio quality")
        
        # Real-time factor comparison
        logger.info(f"\n🏎️  REAL-TIME PERFORMANCE:")
        logger.info(f"   Non-streaming: {ns['realtime_factor']:.2f}x real-time")
        logger.info(f"   Streaming:     {s['realtime_factor']:.2f}x real-time")
        
        # Final verdict
        logger.info(f"\n🧪 HYPOTHESIS TEST RESULT:")
        if time_diff > 0:
            logger.info(f"   ✅ HYPOTHESIS CONFIRMED!")
            logger.info(f"   📈 Streaming is {time_diff:.3f}s ({time_improvement:.1f}%) faster")
            logger.info(f"   🎯 Performance improvement validates streaming architecture")
        else:
            logger.info(f"   ❓ HYPOTHESIS NEEDS INVESTIGATION")
            logger.info(f"   📊 Results suggest overhead or implementation factors")
            logger.info(f"   💡 Consider: text size, network latency, processing pipeline")

async def main():
    """Run the fixed performance comparison"""
    logger.info("🚀 FIXED Performance Comparison: Streaming vs Non-Streaming")
    logger.info(f"🌐 Server: {CHATTERBOX_SERVER_URL}")
    logger.info(f"📝 Text length: {len(COMPARISON_TEXT)} characters")
    logger.info(f"⏱️  Expected synthesis time: ~24 seconds (based on server logs)")
    logger.info(f"🔧 Using 60-second timeouts to handle long synthesis")
    
    comparison = FixedPerformanceComparison()
    
    # Use longer timeout for the session too
    timeout = aiohttp.ClientTimeout(total=90, sock_connect=30, sock_read=60)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        logger.info("📡 Created aiohttp session with extended timeouts")
        
        try:
            # Test non-streaming first
            logger.info(f"\n🔄 Starting NON-STREAMING test at {time.strftime('%H:%M:%S')}")
            ns_success = await comparison.test_non_streaming_with_timeout(session)
            
            if ns_success:
                logger.info("✅ Non-streaming test completed, waiting 10s before streaming test...")
                await asyncio.sleep(10)  # Longer pause to let server recover
                
                logger.info(f"\n🔄 Starting STREAMING test at {time.strftime('%H:%M:%S')}")
                s_success = await comparison.test_streaming_with_timeout(session)
                
                if s_success:
                    comparison.analyze_results()
                else:
                    logger.error("❌ Streaming test failed")
            else:
                logger.error("❌ Non-streaming test failed, skipping streaming test")
            
            logger.info(f"\n📁 Audio files saved to: {OUTPUT_DIR.absolute()}")
            logger.info("🎧 Download and compare the audio files!")
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())