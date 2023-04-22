'''

Contributed by SagsMug. Thank you SagsMug.
https://github.com/oobabooga/text-generation-webui/pull/175

'''

import asyncio
import json
import sys
from typing import AsyncIterator

try:
    import websockets
except ImportError:
    print("Websockets package not found. Make sure it's installed.") 

HOST = 'localhost:5005'
URI = f'ws://{HOST}/api/v1/stream'

async def run(context):
    # Note: the selected defaults change from time to time.
    request = {
        'prompt': context,
        'max_length': 250,
        'do_sample': True,
        'temperature': 1.3,
        'top_p': 0.1,
        'typical': 1,
        'rep_pen': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    async with websockets.connect(URI) as websocket:
        print('connected.')
        await websocket.send(json.dumps(request))
        print('sent request.')

        while True:
            await websocket.drain()
            incoming_data = await websocket.recv()
            print('got incoming data.')
            incoming_data = json.loads(incoming_data)

            match incoming_data['event']:
                case 'text_stream':
                    print('received text.')
                    yield incoming_data['text']
                case 'stream_end':
                    print('received stream end.')
                    return

prompt = "These are the best places to see the cherry blossoms in Seattle:"

async def _get_response_stream(prompt) -> AsyncIterator[str]:
    async for response in run(prompt):
        yield response

async def print_response_stream(prompt):
    async for response in _get_response_stream(prompt):
        print(response, end='')
        sys.stdout.flush() # If we don't flush, we won't see tokens in realtime.

if __name__ == '__main__':
    asyncio.run(print_response_stream(prompt))
