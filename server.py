import asyncio
import websockets
import json
import random

connected_clients = set()

PORT = 12345
ADDRESS = 'localhost'

def parse_message(message):
    data = json.loads(message)
    if 'type' not in data:
        pass
    if data['type'] == 'begin':
        pass
    elif data['type'] == 'select':
        pass


async def send_duel_message(websocket):
    while True:
        message = {
            "candidates": [
                {
                    "qIn": random.uniform(0.0, 1.0),
                    "qOut": random.uniform(0.0, 1.0)
                },
                {
                    "qIn": random.uniform(0.0, 1.0),
                    "qOut": random.uniform(0.0, 1.0)
                }
            ]
        }

        await websocket.send(json.dumps(message))
        await asyncio.sleep(0.5)


async def handle_client(websocket):
    connected_clients.add(websocket)

    duel_task = asyncio.create_task(send_duel_message(websocket))

    try:
        async for messages in websocket:
            # handle incoming messages here if needed
            pass

    except websockets.exceptions.ConnectionClosed:
        pass

    finally:
        duel_task.cancel()
        connected_clients.remove(websocket)

async def main():
    server = await websockets.serve(handle_client, ADDRESS, PORT)
    print(f"WebSocket Server starting on ws://{ADDRESS}:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())


