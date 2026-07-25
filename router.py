from handler.v1 import handle_client as handler_v1
from handler.v2 import handle_client as handler_v2 

async def route_connection(websocket) -> None:
    path = websocket.request.path

    if path == "/v1":
        await handler_v1(websocket)
    elif path == "/v2":
        await handler_v2(websocket)
    else:
        await websocket.close(code=1008, reason="Unknown WS Path")

