import heapq

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from loguru import logger

from snaikenet_client.client_data import ClientGameStateFrame
from snaikenet_client.types import ClientDirection, ClientGridStructure, ClientTileType
from snaikenet_rl_sabrinafair.game_controller import ClientPhase

GRID_LENGTH = 49
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
SCOREBOARD_HEIGHT = 40

COLORS = {
    ClientTileType.EMPTY: (20, 20, 20),
    ClientTileType.WALL: (100, 100, 100),
    ClientTileType.FOOD: (220, 50, 50),
    ClientTileType.SNAKE: (50, 220, 80),
    ClientTileType.OTHER_SNAKE: (80, 120, 220),
}


X_INDEX = 0
Y_INDEX = 1

#####################HELPER FUNCTIONS########################
def get_sqr_eucl_dist(pos1, pos2):
    return (pos1[X_INDEX] - pos2[X_INDEX])**2 + (pos1[Y_INDEX] - pos2[Y_INDEX])**2


class ClientEnv(gym.Env):
    # metadata = {"render_modes": []}
    metadata = {"render_modes": ["human", None]}

    def __init__(self, controller, grid_size=(GRID_LENGTH, GRID_LENGTH), headless=False):
        super().__init__()
        self.controller = controller
        self.headless = headless
        self.h, self.w = grid_size
        self.max_steps = 500
        self.steps = 0
        self.last_state = None
        self.closest_food = self.get_distance_to_food(self.controller.latest_frame)

        # DQN in SB3 expects a discrete action space
        self.action_space = spaces.Discrete(4)
        self.direction = ClientDirection(ClientDirection.WEST)

        # Start simple: integer grid observation
        # self.observation_space = spaces.Box(
        #     low=0,
        #     high=40,
        #     shape=(self.h * self.w + 1,),
        #     dtype=np.float32,
        # )
        #update to be current position (x1,y1), current direction, and distance to closest food (distX, distY)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64
        )
        # print("self.observation_space")
        # print(self.observation_space)
        # print(self.observation_space.shape)
        self.viewport_w = 0
        self.viewport_h = 0
        self.tile_size = 1
        self.grid_offset_x = 0
        self.grid_offset_y = 0
        self.countdown_seconds = 0
        self.running = True
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("SnaikeNET")
            self.font = pygame.font.SysFont("consolas", 18)
            self.big_font = pygame.font.SysFont("consolas", 120)
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.font = None
            self.big_font = None
            self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        state = self.controller.reset_episode(seed=seed)
        self.last_state = state

        obs = self._obs_from_state(self.direction)
        info = {}
        return obs, info

    # def is_valid_direction(self, action_direction, curr_direction):
    #     isValidMove = True
    #     invalid_move = {
    #         ClientDirection.NORTH: ClientDirection.SOUTH,
    #         ClientDirection.SOUTH: ClientDirection.NORTH,
    #         ClientDirection.EAST: ClientDirection.WEST,
    #         ClientDirection.WEST: ClientDirection.EAST,
    #     }
    #     if action_direction == invalid_move[curr_direction]:
    #         isValidMove = False
    #     return isValidMove

    def step(self, action: int):
        
        self.direction = ClientDirection(action)
        prev_seq = self.controller.latest_frame.sequence_number if self.controller.latest_frame is not None else -1

        self.controller.direction_queue.put(self.direction)

        while True:
            ev = self.controller.handler.event_queue.get()

            if ev.kind == "frame":
                frame = ev.frame

                # only return a genuinely newer frame
                if frame.sequence_number > prev_seq:
                    self.controller.old_frame = self.controller.latest_frame
                    self.controller.latest_frame = frame
                    self.steps += 1
            elif ev.kind == "game_end":
                self.game_over = True
            else:
                continue

            reward = self._reward(self.controller.latest_frame, self.controller.old_frame)
            obs = self._obs_from_state(self.direction)
            terminated = not self.controller.latest_frame.is_alive
            truncated = self.steps >= self.max_steps
            info = self._info(self.controller.latest_frame)

            if self.headless:
                return obs, reward, terminated, truncated, info

            # Render
            self.screen.fill((0, 0, 0))

            # print(phase)
            if self.controller.phase == ClientPhase.WAITING:
                if self.viewport_w > 0 and self.viewport_h > 0:
                    self.controller.render_empty_grid(
                        self.screen,
                        self.font,
                        self.viewport_w,
                        self.viewport_h,
                        self.tile_size,
                        self.grid_offset_x,
                        self.grid_offset_y,
                    )
                else:
                    self.controller.render_waiting(self.screen, self.font)
            elif self.controller.phase == ClientPhase.COUNTDOWN:
                if self.viewport_w > 0 and self.viewport_h > 0:
                    self.controller.render_empty_grid(
                        self.screen,
                        self.font,
                        self.viewport_w,
                        self.viewport_h,
                        self.tile_size,
                        self.grid_offset_x,
                        self.grid_offset_y,
                    )
                self.render_countdown(self.screen, self.big_font, self.countdown_seconds)
            elif self.controller.phase == ClientPhase.PLAYING and self.controller.latest_frame is not None:
                # Recompute layout from actual frame grid in case it differs
                grid_w = len(self.controller.latest_frame.grid_data)
                grid_h = len(self.controller.latest_frame.grid_data[0]) if grid_w else 0
                if grid_w != self.viewport_w or grid_h != self.viewport_h:
                    self.update_grid_layout(grid_w, grid_h)
                self.controller.render_frame(
                    self.screen,
                    self.controller.latest_frame,
                    self.font,
                    self.tile_size,
                    self.grid_offset_x,
                    self.grid_offset_y,
                )
            # elif self.controller.phase == ClientPhase.GAME_OVER:
            #     logger.info("GAME END")
            pygame.display.flip()
            self.clock.tick(60)


            return obs, reward, terminated, truncated, info

    def close(self):
        self.controller.close()


    def get_matrix_from_state(self, state: ClientGameStateFrame):
        return np.array(
            [[int(cell) for cell in row] for row in state.grid_data],
            dtype=np.float32,
        )
    
    def _obs_from_state(self, curr_direction):
        curr_location = self.get_curr_location()
        dist_to_target = self.get_distances_to_food(self.controller.latest_frame)

        obs = np.hstack([curr_location, curr_direction, dist_to_target])

        return obs
    
    def _info(self, frame):
        return {
            "sequence_number": frame.sequence_number,
            "length": frame.player_length,
            "kills": frame.num_kills,
            "alive": frame.is_alive,
            "spectating": frame.is_spectating,
        }

    def _reward(self, new_state: ClientGameStateFrame, old_state: ClientGameStateFrame):

        if not new_state.is_alive:
            return -10.0
        
        reward = 0.0
        
        #positive reward for length of snake
        old_len = self.get_snake_length(old_state)
        new_len = self.get_snake_length(new_state)
        if new_len > old_len:
            reward += 20.0
        
        old_dist, _ = self.get_distance_to_food(old_state)
        new_dist, max_dist = self.get_distance_to_food(new_state)
        # less penalty the closer to food
        # reward += -3 * self.norm_dist(dist)
        # reward += 3 *(self.norm_dist(old_dist) - self.norm_dist(new_dist))
        reward += -1 * (new_dist / max_dist) #normalized


        #small step penalty
        reward -= 0.05
        return reward
    

    def get_distance_to_food(self, state: ClientGameStateFrame):

        #get indices where there is food
        priority_queue = []
        snake_row= GRID_LENGTH / 2
        snake_col = GRID_LENGTH / 2
        if state:
            rows, cols = np.where(self.get_matrix_from_state(state) == ClientTileType.FOOD)

            for i in range(len(rows)):

                p1 = np.array([rows[i], cols[i]])
                p2 = np.array([snake_row, snake_col])
                # distance = np.linalg.norm(p1 - p2)
                distance = get_sqr_eucl_dist(p1, p2)
                heapq.heappush(priority_queue, (distance, (rows[i], cols[i])))

        p1 = np.array([0, 0])
        p2 = np.array([snake_row, snake_col])
        max_distance = get_sqr_eucl_dist(p1, p2)

        if priority_queue:
            dist, (r, c) = heapq.heappop(priority_queue)
        else:
            dist = max_distance
        priority_queue.clear()

        self.closest_food = dist
        return dist, max_distance
    
    def get_distances_to_food(self, state: ClientGameStateFrame):

        #get indices where there is food
        priority_queue = []
        snake_row= GRID_LENGTH / 2
        snake_col = GRID_LENGTH / 2
        if state:
            rows, cols = np.where(self.get_matrix_from_state(state) == ClientTileType.FOOD)

            for i in range(len(rows)):

                p1 = np.array([rows[i], cols[i]])
                p2 = np.array([snake_row, snake_col])
                distance = np.linalg.norm(p1 - p2)
                heapq.heappush(priority_queue, (distance, (rows[i], cols[i])))

        p1 = np.array([0, 0])
        p2 = np.array([snake_row, snake_col])
        dist = np.linalg.norm(p1 - p2)

        if priority_queue:
            dist, (r, c) = heapq.heappop(priority_queue)
            p1 = np.array([r, c])

        priority_queue.clear()

        self.closest_food = dist
        return (p1 - p2)
    
    def get_curr_location(self):

        snake_row= GRID_LENGTH / 2
        snake_col = GRID_LENGTH / 2

        return np.array([snake_row, snake_col])

    def norm_dist(self, dist):

        snake_row= GRID_LENGTH / 2
        snake_col = GRID_LENGTH / 2
        p1 = np.array([0, 0])
        p2 = np.array([snake_row, snake_col])
        # max_distance = np.linalg.norm(p1 - p2)

        return dist/max_distance

    def get_snake_length(self, state: ClientGameStateFrame):
        if state:
            rows, cols = np.where(self.get_matrix_from_state(state) == ClientTileType.SNAKE)
            return len(rows)
        else:
            return 1

    def _terminated(self, state):
        return not self.controller.latest_frame.is_alive
    
    def update_grid_layout(self, vw: int, vh: int):
        # nonlocal viewport_w, viewport_h, tile_size, grid_offset_x, grid_offset_y
        self.viewport_w, viewport_h = vw, vh
        self.tile_size = self.controller.compute_tile_size(vw, vh)
        self.grid_pixel_w = vw * self.tile_size
        self.grid_pixel_h = vh * self.tile_size
        self.grid_offset_x = (WINDOW_WIDTH - self.grid_pixel_w) // 2
        self.grid_offset_y = (
            SCOREBOARD_HEIGHT + (WINDOW_HEIGHT - SCOREBOARD_HEIGHT - self.grid_pixel_h) // 2
        )