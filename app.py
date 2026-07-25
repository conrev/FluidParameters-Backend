import asyncio
import websockets

from router import route_connection

connected_clients = set()

PORT = 12345
ADDRESS = "0.0.0.0"

async def main():
    server = await websockets.serve(route_connection, ADDRESS, PORT)
    print(f"WebSocket Server starting on ws://{ADDRESS}:{PORT}")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
