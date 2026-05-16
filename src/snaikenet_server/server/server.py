import asyncio
import ctypes
import json
import socket
import sys
import time
from uuid import uuid4

from loguru import logger

from snaikenet_protocol.protocol import ServerCodec
from snaikenet_server.game.game_state import PlayerView
from snaikenet_server.server.connected_clients import ConnectedClients
from snaikenet_server.server.server_event_handler import (
    SnaikenetServerEventHandler,
    DefaultSnaikenetServerEventHandler,
)

SIO_UDP_CONNRESET = 0x9800000C


def _disable_udp_connreset(sock: socket.socket):
    """Disable SIO_UDP_CONNRESET via WSAIoctl (sock.ioctl rejects this command on newer Python)."""
    ws2_32 = ctypes.WinDLL("ws2_32", use_last_error=True)
    enable = ctypes.c_ulong(0)
    bytes_returned = ctypes.c_ulong(0)

    result = ws2_32.WSAIoctl(
        sock.fileno(),
        SIO_UDP_CONNRESET,
        ctypes.byref(enable),
        ctypes.sizeof(enable),
        None,
        0,
        ctypes.byref(bytes_returned),
        None,
        None,
    )
    if result != 0:
        err = ctypes.get_last_error()
        raise OSError(f"WSAIoctl(SIO_UDP_CONNRESET) failed: WinError {err}")


class SnaikenetServer:
    """
    Server for Snaikenet game
    TCP Registration:
        1) Client: sends registration request over TCP
        2) Server: generates UUID and ACKs back to client with UUID and UDP port to use for hole punching
        3) Client: sends a UDP datagram containing the UUID to the server's UDP port to complete registration
        4) Server: server reads NAT-mapped addr and registers the client with the UUID and NAT-mapped addr for future communication

    TCP Reconnection:
        1) Client: sends reconnection request over TCP with UUID
        2) Server: validates UUID, responds with UDP port
        3) Client: Sends UDP datagram containing the UUID
        4) Server: updates client's NAT-mapped addr for future communication
    """

    def __init__(
        self,
        host="localhost",
        tcp_port=8888,
        udp_port=8888,
        event_handler: SnaikenetServerEventHandler = DefaultSnaikenetServerEventHandler(),
        client_timeout_seconds: float = 20,
        preset_player_ids: list[str] = [],
    ):
        self._host: str = host
        self._tcp_port: int = tcp_port
        self._udp_port: int = udp_port
        self._connected_clients: ConnectedClients = ConnectedClients()
        # Temporary storage for pending hole-punching requests: UUID -> Future that will be set with (host, port) when UDP datagram is received
        self._pending_clients: dict[str, asyncio.Future[tuple[str, int]]] = {}
        self._event_handler: SnaikenetServerEventHandler = event_handler
        self._tcp_server: asyncio.Server | None = None
        self._udp_transport: asyncio.DatagramTransport | None = None
        self._clean_idle_clients_task = None
        self._client_timeout_seconds = client_timeout_seconds
        self._preset_player_ids = list(
            reversed(preset_player_ids)
        )  # reverse so that pop can be used to just remove the top one

    def get_host(self) -> str:
        return self._host

    def get_tcp_port(self) -> int:
        return self._tcp_port

    def get_udp_port(self) -> int:
        return self._udp_port

    def broadcast_game_start(self, viewport_size: tuple[int, int]):
        if self._udp_transport is None:
            self._log_udp_not_initialized()
            return
        for dest in self._connected_clients.get_client_addrs():
            self._udp_transport.sendto(
                ServerCodec.encode_game_start(viewport_size), dest
            )

    def broadcast_game_restart(self):
        if self._udp_transport is None:
            self._log_udp_not_initialized()
            return
        for dest in self._connected_clients.get_client_addrs():
            self._udp_transport.sendto(ServerCodec.encode_game_restart(), dest)

    def broadcast_game_end(self):
        logger.info("Broadcasting game end...")
        if self._udp_transport is None:
            self._log_udp_not_initialized()
            return
        for dest in self._connected_clients.get_client_addrs():
            self._udp_transport.sendto(ServerCodec.encode_game_end(), dest)

    async def broadcast_game_state_frames(
        self, client_frames: dict[str, PlayerView], sequence_number: int
    ):
        start_broadcast_time = time.perf_counter()
        for uuid, frame in client_frames.items():
            self._broadcast_game_state_frame(uuid, frame, sequence_number)
        logger.debug(
            f"{(time.perf_counter() - start_broadcast_time) * 1000:.3f}ms to broadcast game state frames to all clients"
        )

    def _broadcast_game_state_frame(
        self, client_id: str, client_frame: PlayerView, sequence_number: int
    ):
        if self._udp_transport is None:
            self._log_udp_not_initialized()
            return
        dest = self._connected_clients.get_client_by_id(client_id)
        if dest is not None:
            encoded = ServerCodec.encode_player_game_state(
                client_frame,
                sequence_number,
            )
            self._udp_transport.sendto(encoded, dest.get_addr())

    @staticmethod
    def _log_udp_not_initialized():
        logger.error(f"UDP transport not initialized. Make sure start method is called")

    async def start(self, clean_idle_clients: bool = True):
        loop = asyncio.get_running_loop()

        self._tcp_server = _tcp_server = await asyncio.start_server(
            self._handle_registration, self._host, self._tcp_port
        )
        self._tcp_port = _tcp_server.sockets[0].getsockname()[1]
        logger.info(f"TCP server started on {self._host}:{self._tcp_port}")

        await loop.create_datagram_endpoint(
            lambda: self._UdpProtocol(self), local_addr=(self._host, self._udp_port)
        )
        # noinspection PyUnresolvedReferences
        self._udp_port = self._udp_transport.get_extra_info("socket").getsockname()[1]
        logger.info(f"UDP server started on {self._host}:{self._udp_port}")

        if clean_idle_clients:
            self._clean_idle_clients_task = loop.create_task(
                self._clean_idle_clients_loop()
            )

    def get_clients_str(self) -> str:
        return "Clients connected: " + ", ".join(
            f"{client.get_id()} at {client.get_addr()}"
            for client in self._connected_clients.get_clients()
        )

    def get_client_ids(self) -> list[str]:
        return [client.get_id() for client in self._connected_clients.get_clients()]

    async def serve_forever(self):
        if self._tcp_server is None:
            raise RuntimeError("Server not started yet")
        async with self._tcp_server:
            await self._tcp_server.serve_forever()

    async def stop(self):
        self.broadcast_game_end()
        if self._clean_idle_clients_task:
            logger.info("Stopping clean idle clients task...")
            self._clean_idle_clients_task.cancel()
        if self._udp_transport:
            logger.info("Stopping UDP server...")
            self._udp_transport.close()
        if self._tcp_server:
            logger.info("Stopping TCP server...")
            self._tcp_server.close()
            await self._tcp_server.wait_closed()
        logger.info("SnaikeNET server stopped.")

    async def _handle_registration(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ):
        peer = writer.get_extra_info("peername")
        logger.info(f"Received connection from {peer}")

        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not raw:
                return

            msg = json.loads(raw.decode().strip())
            req_type = msg.get("type")

            match req_type:
                case "new":
                    # pop preset player_ids and then use uuid4 if there are no more available
                    client_id = (
                        self._preset_player_ids.pop()
                        if self._preset_player_ids
                        else str(uuid4())
                    )
                    self._connected_clients.remove_client_by_id(
                        client_id
                    )  # Clear any old registration with same UUID just in case, should be very unlikely
                    logger.debug(
                        f"New client registering with UUID {client_id} from TCP {peer}"
                    )
                case "reconnect":
                    client_id = msg.get("uuid")
                    if not self._connected_clients.has_client_id(client_id):
                        writer.write(
                            ServerCodec.error_response("Unknown UUID: Cannot reconnect")
                        )
                        await writer.drain()
                        return

                    self._connected_clients.remove_client_by_id(client_id)
                    logger.info(
                        f"Client with UUID {client_id} is reconnecting from {peer}"
                    )
                case _:
                    writer.write(ServerCodec.error_response("Invalid request type"))
                    await writer.drain()
                    return

            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            self._pending_clients[client_id] = fut

            writer.write(
                ServerCodec.udp_hole_punch_success_request(client_id, self._udp_port)
            )
            logger.debug(
                f"Sending hole punch request to client {client_id} at TCP {peer}"
            )
            await writer.drain()

            try:
                external_addr: tuple[str, int] = await asyncio.wait_for(
                    fut, timeout=10.0
                )
            except asyncio.TimeoutError:
                pending_client = self._pending_clients.pop(client_id, None)
                if pending_client:
                    await pending_client

                if req_type == "new":
                    self._connected_clients.remove_client_by_id(
                        client_id
                    )  # Clear any registration for this client since hole punch failed
                logger.warning(f"Hole punch timed out for client {client_id}")
                writer.write(ServerCodec.error_response("Hole punch timed out"))
                await writer.drain()
                return

            # At this point, we have already handled if this is a reconnect and cleared the old address, so we can just register the new address
            self._connected_clients.register_client(client_id, external_addr)
            if req_type == "new":
                is_spectator: bool = msg.get("spectator")
                if not isinstance(is_spectator, bool):
                    is_spectator = False
                self._event_handler.on_new_client_connect(client_id, is_spectator)

            logger.info(
                f"Client {client_id} hole-punched successfully with external address {external_addr}. Sending registration success response to TCP {peer}"
            )
            writer.write(ServerCodec.udp_hole_punch_success_response())
            await asyncio.wait_for(writer.drain(), timeout=5.0)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Invalid registration request from {peer}: {e}")
            writer.write(ServerCodec.error_response("Invalid registration request"))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    class _UdpProtocol(asyncio.DatagramProtocol):
        def __init__(self, server: "SnaikenetServer"):
            self._server = server

        def connection_made(self, transport: asyncio.DatagramTransport):
            self._server._udp_transport = transport

            # fixes Windows specific bug that caused the socket to stop reading
            # whenever it got a SIO_UDP_CONNRESET request that can be induced from the client
            if sys.platform == "win32":
                logger.info(
                    "Disabling SIO_UDP_CONNRESET on Windows UDP socket to prevent it from being disabled by certain UDP packets from clients"
                )
                _disable_udp_connreset(transport.get_extra_info("socket"))

        def error_received(self, exc: Exception):
            logger.warning(f"UDP transport error (ignored): {exc}")

        def datagram_received(self, data: bytes, addr: tuple[str, int]):
            try:
                try:
                    msg = data.decode().strip()
                    msg_json = json.loads(msg)
                except UnicodeDecodeError, json.JSONDecodeError:
                    logger.warning(f"Ignoring non-JSON UDP packet from {addr}")
                    return

                msg_type = msg_json.get("type")

                if msg_type == "hole_punch":
                    client_id = msg_json.get("uuid")
                    # Hole-punch ping
                    if client_id in self._server._pending_clients:
                        logger.info(
                            f"Received hole punch ping from {addr} for client {client_id}"
                        )
                        fut = self._server._pending_clients.pop(client_id)
                        if not fut.done():
                            fut.set_result(addr)
                        return
                    elif self._server._connected_clients.has_client_id(client_id):
                        logger.debug(
                            f"Received hole punch for existing client {client_id} from {addr}. Ignoring since client is already registered."
                        )
                        return
                elif msg_type == "direction":
                    try:
                        client = self._server._connected_clients.get_client_by_addr(
                            addr
                        )
                        if client is None:
                            raise ValueError(
                                f"No registered client with address {addr}"
                            )
                        self._server._connected_clients.touch_client_by_id(
                            client.get_id()
                        )
                        decoded_direction = ServerCodec.decode_direction(data)
                        if decoded_direction is None:
                            raise ValueError(
                                f"Invalid direction message from {addr}: {msg}"
                            )
                        self._server._event_handler.on_receive_direction(
                            client.get_id(), decoded_direction
                        )
                    except ValueError as e:
                        logger.debug(e)
                elif msg_type == "heartbeat":
                    self._server._connected_clients.touch_client_by_addr(addr)

            except Exception as e:
                logger.warning(f"UDP transport error (ignored): {e}")

    def broadcast_game_about_to_start(self, seconds_until_start: int):
        if self._udp_transport is None:
            self._log_udp_not_initialized()
            return
        for client in self._connected_clients.get_clients():
            dest = client.get_addr()
            self._udp_transport.sendto(
                ServerCodec.encode_game_about_to_start(seconds_until_start), dest
            )

    async def wait_start_game_timer(self, seconds_until_start: int):
        for seconds in range(seconds_until_start, 0, -1):
            logger.info(f"Game starting in {seconds} seconds...\n")
            # Broadcast the "game about to start" message multiple
            # times to increase chance of delivery to clients since UDP is unreliable
            for _ in range(10):
                self.broadcast_game_about_to_start(seconds)
                await asyncio.sleep(0.1)

    async def _clean_idle_clients_loop(self):
        while True:
            await asyncio.sleep(1)
            current_time = asyncio.get_running_loop().time()
            clients_to_remove = []
            for client in self._connected_clients.get_clients():
                if current_time - client.get_last_seen() > self._client_timeout_seconds:
                    clients_to_remove.append(client)
            for client in clients_to_remove:
                addr = client.get_addr()
                client_id = client.get_id()
                if self._udp_transport is not None:
                    self._udp_transport.sendto(ServerCodec.encode_game_end(), addr)
                self._event_handler.on_client_disconnect(client_id)
                self._connected_clients.remove_client_by_id(client_id)
