from fastapi import WebSocket
from typing import Set


class WebSocketServerSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WebSocketServerSingleton, cls).__new__(cls)
            cls._instance.connections = set()
        return cls._instance

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.connections.discard(websocket)

    async def broadcast(self, message: str):
        for conn in self.connections:
            await conn.send_text(message)

async def send_ws_message(message: str):
    ws_server = WebSocketServerSingleton()
    await ws_server.broadcast(message)
