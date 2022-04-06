import asyncio
import json
import logging
import websockets


logging.basicConfig()

USERS = set()


# 用户数量事件
def users_event():
    return json.dumps({'type': 'users', 'count': len(USERS)})


async def handler(websocket):
    global USERS
    try:
        USERS.add(websocket)
        websockets.broadcast(USERS, users_event())

        async for message in websocket:
            event = json.loads(message)
            if event['type'] in ['model_ready', 'model_value']:
                websockets.broadcast(USERS, message)
            else:
                logging.error('unsupported event: %s', event)

    finally:
        USERS.remove(websocket)
        websockets.broadcast(USERS, users_event())


async def main():
    async with websockets.serve(handler, "", 8765):
        await asyncio.Future()  # fun forever


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())

