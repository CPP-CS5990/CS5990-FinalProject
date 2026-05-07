import asyncio


class SnaikenetAdminClient:
    def __init__(self, host: str = "localhost", port: int = 8889):
        self.host = host
        self.port = port

    async def send_command(self, command: str) -> dict:
        reader, writer = await asyncio.open_connection(self.host, self.port)

        writer.write(f"{command}\n".encode())
        await writer.drain()

        raw = await reader.readline()

        writer.close()
        await writer.wait_closed()

        return {"raw": raw.decode().strip()}

    async def start_game(self) -> dict:
        return await self.send_command("start")

    async def restart_game(self) -> dict:
        return await self.send_command("restart")
