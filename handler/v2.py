import websockets
import json
from optim.PBO import PreferentialBOSession

connected_clients = set()

async def handle_client(websocket):
    connected_clients.add(websocket)
    session = None  # per-connection state
    await websocket.send(json.dumps({"type": "connected", "message": "PBO Backend v2"}))

    try:
        async for raw in websocket:
            data = json.loads(raw)  # parse once, here
            msg_type = data.get("type", "duel")
            if msg_type == "init":
                session = PreferentialBOSession(
                    json.loads(data["parameters"]),
                    n_init=10,
                    n_iterations=12,
                    warmup="sobol",
                )
                response = await session.start_async()
            elif msg_type == "duel":
                if session is None:
                    response = {"type": "error", "message": "send 'init' before 'duel'"}
                else:
                    try:
                        response = await session.submit_preference_async(
                            data["duelId"], data["choice"]
                        )
                    except (ValueError, RuntimeError) as exc:
                        response = {"type": "error", "message": str(exc)}
            else:
                response = {"type": "error", "message": f"unknown type: {msg_type!r}"}

            if response.get("type") == "result":
                print("One BO Loop Completed, Returning result")
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)

