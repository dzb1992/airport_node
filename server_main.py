# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import asyncio
import websockets


JOIN = {}


async def handler(websocket):
    if not JOIN:
        connected = {websocket}
        JOIN['clients'] = connected
    else:
        connected = JOIN['clients']
        connected.add(websocket)

    print(connected)

    while True:
        # path标识请求路径，可以来自定义需求
        message = await websocket.recv()
        print(f"client: {message}")

        greeting = f"{message}"

        # await websocket.send(greeting)
        websockets.broadcast(connected, greeting)
        print(f"server: {greeting}")


async def main():
    async with websockets.serve(handler, "", 8765):
        await asyncio.Future()  # fun forever


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())

