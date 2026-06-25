import asyncio
import websockets
import json
from .optim import *

connected_clients = set()

PORT = 12345
ADDRESS = 'localhost'

async def handle_client(websocket):
    connected_clients.add(websocket)
    try:
        async for messages in websocket:
            pass
            # print(f"[>] Received: {messages}")
            # await websocket.send(messages)
            # print(f"[<] Sent: {messages}")


    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Remove the client from the set of connected clients
        connected_clients.remove(websocket)

def parse_message(message):
    data = json.loads(message)
    if 'type' not in data:
        pass
    if data['type'] == 'begin':
        pass
    elif data['type'] == 'select':
        pass

async def main():
    server = await websockets.serve(handle_client, ADDRESS, PORT)
    print(f"WebSocket Server starting on ws://{ADDRESS}:{PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())


