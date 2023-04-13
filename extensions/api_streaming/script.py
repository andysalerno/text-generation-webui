import json
import asyncio
from websockets.server import serve
from threading import Thread

from modules import shared
from modules.text_generation import encode, generate_reply

PATH = '/api/v1/stream'

params = {
    'port': 5000,
}


async def _handle_connection(websocket, path):

    if path != PATH:
        return

    async for message in websocket:
        message = json.loads(message)

        prompt = message['prompt']

        prompt_lines = [k.strip() for k in prompt.split('\n')]

        max_context = message.get('max_context_length', 2048)

        while len(prompt_lines) >= 0 and len(encode('\n'.join(prompt_lines))) > max_context:
            prompt_lines.pop(0)

        prompt = '\n'.join(prompt_lines)
        generate_params = {
            'max_new_tokens': int(message.get('max_length', 200)),
            'do_sample': bool(message.get('do_sample', True)),
            'temperature': float(message.get('temperature', 0.5)),
            'top_p': float(message.get('top_p', 1)),
            'typical_p': float(message.get('typical', 1)),
            'repetition_penalty': float(message.get('rep_pen', 1.1)),
            'encoder_repetition_penalty': 1,
            'top_k': int(message.get('top_k', 0)),
            'min_length': int(message.get('min_length', 0)),
            'no_repeat_ngram_size': int(message.get('no_repeat_ngram_size', 0)),
            'num_beams': int(message.get('num_beams', 1)),
            'penalty_alpha': float(message.get('penalty_alpha', 0)),
            'length_penalty': float(message.get('length_penalty', 1)),
            'early_stopping': bool(message.get('early_stopping', False)),
            'seed': int(message.get('seed', -1)),
            'add_bos_token': bool(message.get('add_bos_token', True)),
            'truncation_length': int(message.get('truncation_length', 2048)),
            'custom_stopping_strings': [],
            'ban_eos_token': bool(message.get('ban_eos_token', False))
        }

        generator = generate_reply(
            prompt,
            generate_params,
            stopping_strings=message.get('stopping_strings', []),
        )

        # As we stream, only send the new bytes.
        skip_index = len(prompt)
        message_num = 0

        stop_requested = False

        for a in generator:
            if stop_requested:
                return

            async with asyncio.timeout(0.01):
                received = await websocket.recv()
                received = json.loads(received)
                print('stop requested')
                stop_requested = True

            to_send = ''
            if isinstance(a, str):
                to_send = a[skip_index:]
            else:
                to_send = a[0][skip_index:]

            await websocket.send(json.dumps({
                'event': 'text_stream',
                'message_num': message_num,
                'text': to_send
            }))

            skip_index += len(to_send)
            message_num += 1

        await websocket.send(json.dumps({
            'event': 'stream_end',
            'message_num': message_num
        }))


async def get_next_prediction():
    pass


async def _run(host: str, port: int):
    async with serve(_handle_connection, host, port):
        await asyncio.Future()  # run forever


def _run_server():
    server_addr = (
        '0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    print(
        f'Starting KoboldAI compatible streaming api at ws://{server_addr[0]}:{server_addr[1]}{PATH}')
    asyncio.run(_run(host=server_addr[0], port=server_addr[1]))


def setup():
    Thread(target=_run_server, daemon=True).start()
