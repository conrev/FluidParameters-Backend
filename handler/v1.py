
import websockets
import json
from optim.PBO import PreferentialBOSession, PARAM_SPACE

connected_clients = set()

async def handle_client(websocket):
    connected_clients.add(websocket)

    # warmup: "sobol" (default, space-filling) | "lhs" | "random"
    session = PreferentialBOSession(
        PARAM_SPACE, n_init=10, n_iterations=12, warmup="sobol"
    )
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
