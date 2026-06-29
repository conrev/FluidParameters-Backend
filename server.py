import asyncio
import websockets
import json
import random

from optim.PBO import PreferentialBOSession, PARAM_SPACE

connected_clients = set()

PORT = 12345
ADDRESS = "localhost"


async def handle_client(websocket):
    connected_clients.add(websocket)

    session = PreferentialBOSession(PARAM_SPACE, n_init=10, n_iterations=12)
    msg = await session.start_async()
    await websocket.send(json.dumps(msg))

    try:
        async for messages in websocket:
            data = json.loads(messages)
            msg = await session.submit_preference_async(data["duelId"], data["choice"])
            await websocket.send(json.dumps(msg))
            if msg["type"] == "result":
                print("One BO Loop Completed, Returning result")

    except websockets.exceptions.ConnectionClosed:
        pass

    finally:
        connected_clients.remove(websocket)


async def main():
    server = await websockets.serve(handle_client, ADDRESS, PORT)
    print(f"WebSocket Server starting on ws://{ADDRESS}:{PORT}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
