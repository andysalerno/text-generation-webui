'''

Contributed by SagsMug. Thank you SagsMug.
https://github.com/oobabooga/text-generation-webui/pull/175

'''

import asyncio
import json
import websockets
import sys


async def run(prompt):
    server = "127.0.0.1"
    params = {
        'max_new_tokens': 200,
        'do_sample': True,
        'temperature': 0.5,
        'top_p': 0.9,
        'typical_p': 1,
        'repetition_penalty': 1.05,
        'encoder_repetition_penalty': 1.0,
        'top_k': 0,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
    }

    params['prompt'] = prompt

    async with websockets.connect(f"ws://{server}:5000/api/v1/stream") as websocket:
        await websocket.send(json.dumps(params))

        while incoming_data := json.loads(await websocket.recv()):
            match incoming_data['event']:
                case 'text_stream':
                    yield incoming_data['text']
                case 'stream_end':
                    return

prompt = "These are the best places to see the cherry blossoms in Seattle: "


async def get_result():
    print(prompt, end='')

    async for response in run(prompt):
        # Print intermediate steps
        print(response, end='')

        # Flush to see realtime generation. Disable and the runtime will most likely decide to flush line-by-line.
        sys.stdout.flush() 

    # Print final result
    print(response)

asyncio.run(get_result())
