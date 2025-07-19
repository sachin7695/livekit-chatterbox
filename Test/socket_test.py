#!/usr/bin/env python3
"""
Simple WebSocket test to verify Chatterbox server connectivity
"""

import asyncio
import json
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_URL = "ws://46.18.108.33:8765"

async def test_direct_websocket():
    """Test direct WebSocket connection to Chatterbox server"""
    logger.info(f"Testing direct WebSocket connection to {SERVER_URL}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(SERVER_URL) as ws:
                logger.info("✅ WebSocket connection established!")
                
                # Send a simple synthesis request
                request = {
                    "type": "synthesize",
                    "text":
                    """
                    This is a comprehensive test of the streaming capabilities of the Chatterbox TTS system. 
                    We will test how well it handles longer texts with multiple sentences. 
                    The system should be able to process this text incrementally and provide smooth audio output. 
                    This test will help us verify that the integration between the LiveKit plugin and the Chatterbox server is working correctly.
                    """,
                    "request_id": "test_001",
                    "params": {
                        "exaggeration": 0.5,
                        "temperature": 0.8,
                        "cfg_weight": 0.5
                    }
                }
                
                logger.info("Sending synthesis request...")
                await ws.send_str(json.dumps(request))
                
                # Wait for responses
                response_count = 0
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        logger.info(f"Received: {data.get('type')} - {list(data.keys())}")
                        
                        if data.get('type') == 'synthesis_complete':
                            logger.info("✅ Synthesis completed successfully!")
                            break
                        elif data.get('type') == 'error':
                            logger.error(f"❌ Server error: {data.get('error')}")
                            break
                        
                        response_count += 1
                        if response_count > 10:  # Safety limit
                            logger.warning("Too many responses, breaking")
                            break
                    else:
                        logger.info(f"Non-text message: {msg.type}")
                        if msg.type == aiohttp.WSMsgType.ERROR:
                            break
                
                logger.info("Test completed")
                
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        logger.error(f"Make sure your Chatterbox server is running on {SERVER_URL}")

if __name__ == "__main__":
    asyncio.run(test_direct_websocket())