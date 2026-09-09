import websockets
import json
from optim.PBO import PreferentialBOSession
from study.recorder import make_recorder

connected_clients = set()


async def handle_client(websocket):
    connected_clients.add(websocket)
    session = None  # per-connection state
    recorder = None
    await websocket.send(json.dumps({"type": "connected", "message": "PBO Backend v2"}))

    try:
        async for raw in websocket:
            data = json.loads(raw)  # parse once, here
            msg_type = data.get("type", "duel")
            if msg_type == "init":
                n_init = data.get("n_init", 10)
                n_iterations = data.get("n_bo", 12)
                # Study metadata is optional: {"study": {"participant": "P01", ...}} or
                # a flat "participantId". Sessions are recorded either way (an unlabelled
                # session is still analysable); set PBO_DISABLE_SESSION_LOG=1 to opt out.
                recorder = make_recorder(data.get("study") or {"participantId":
                                                               data.get("participantId")})
                session = PreferentialBOSession(
                    json.loads(data["parameters"]),
                    n_init=n_init,
                    n_iterations=n_iterations,
                    warmup="sobol",
                    recorder=recorder,
                    # Study options, both session-level and both off by default:
                    #   randomiseSides — shuffle A/B so a side bias can't masquerade as a
                    #                    preference for the incumbent (always option B otherwise).
                    #   catchEvery     — re-present an earlier pair, sides swapped, every N scored
                    #                    comparisons. Measurement only: never enters the model and
                    #                    never consumes budget, but does cost the participant time.
                    randomise_sides=bool(data.get("randomiseSides", False)),
                    catch_every=int(data.get("catchEvery", 0) or 0),
                )
                response = await session.start_async()
            elif msg_type == "duel":
                if session is None:
                    response = {"type": "error", "message": "send 'init' before 'duel'"}
                else:
                    try:
                        # "client" carries optional interface telemetry (decision time, viewpoint
                        # switches, playback events...). Recorded verbatim; never used by the optimiser.
                        response = await session.submit_preference_async(
                            data["duelId"], data["choice"], data.get("client")
                        )
                    except (ValueError, RuntimeError) as exc:
                        if recorder is not None:
                            recorder.note("client_error", message=str(exc))
                        response = {"type": "error", "message": str(exc)}
            else:
                response = {"type": "error", "message": f"unknown type: {msg_type!r}"}

            if response.get("type") == "result":
                print("One BO Loop Completed, Returning result")
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosed:
        # An abandoned session is study data too: record why it ended before finalising.
        if recorder is not None:
            recorder.close("disconnected")
    finally:
        if recorder is not None:
            recorder.close("closed")
        connected_clients.discard(websocket)
