from ast import List
import json
import asyncio
from websockets.server import serve
from threading import Thread
from collections import deque

from modules import shared
from modules.text_generation import encode, generate_reply

PATH = '/api/v1/stream'

params = {
    'port': 5000,
}

tokens_queue = deque()
incoming_generate_queue = deque()
stop_requested = False


async def _handle_connection(websocket, path):
    global incoming_generate_queue
    global stop_requested

    if path != PATH:
        return

    while True:
        message = await websocket.recv()

        print(f'got message: {message}')

        message = json.loads(message)

        prompt = message.get('prompt', '')

        stop_requested = message.get('stop_requested', False)

        if stop_requested:
            await websocket.close()
            return

        if prompt == '':
            continue

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
            'custom_stopping_strings': message.get('custom_stopping_strings', []),
            'ban_eos_token': bool(message.get('ban_eos_token', False))
        }

        # tell the generator to start generating
        incoming_generate_queue.append((generate_params, prompt, websocket))


async def text_generator():
    global stop_requested
    global incoming_generate_queue

    print('starting text generator.')

    while True:
        if not any(incoming_generate_queue):
            await asyncio.sleep(0.1)
            continue

        print('beginning to generate')
        (generate_params, prompt, websocket) = incoming_generate_queue.popleft()

        generator = generate_reply(
            prompt,
            generate_params,
            stopping_strings=generate_params.get(
                'stopping_strings', generate_params['custom_stopping_strings']),
        )

        # As we stream, only send the new bytes.
        skip_index = len(prompt)
        message_num = 0

        try:
            for a in generator:
                if stop_requested:
                    stop_requested = False
                    break

                if websocket.closed:
                    print('closed connection.')
                    break

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
        except:
            pass

        print('done generating.')


async def _run(host: str, port: int):
    async with serve(_handle_connection, host, port):
        await asyncio.Future()  # run forever


def _run_generator():
    asyncio.run(text_generator())


def _run_server():
    server_addr = (
        '0.0.0.0' if shared.args.listen else '127.0.0.1', params['port'])
    print(
        f'Starting KoboldAI compatible streaming api at ws://{server_addr[0]}:{server_addr[1]}{PATH}')
    asyncio.run(_run(host=server_addr[0], port=server_addr[1]))


def setup():
    Thread(target=_run_server, daemon=True).start()
    Thread(target=_run_generator, daemon=True).start()
