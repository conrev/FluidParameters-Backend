import asyncio
import websockets
import json

from optim.PBO import PreferentialBOSession, PARAM_SPACE

connected_clients = set()

PORT = 12345
ADDRESS = "localhost"


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


async def handle_client_request(websocket):
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


async def main():
    server = await websockets.serve(handle_client_request, ADDRESS, PORT)
    print(f"WebSocket Server starting on ws://{ADDRESS}:{PORT}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
