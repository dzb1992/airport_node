# WS server example

import asyncio
import websockets


async def node_detect(websocket, path):
    while True:

        # path标识请求路径，可以来自定义需求
        name = await websocket.recv()
        print(f"< {name}")

        greeting = f"node: {name}"

        await websocket.send(greeting)
        print(f"> {greeting}")


start_server = websockets.serve(node_detect, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
