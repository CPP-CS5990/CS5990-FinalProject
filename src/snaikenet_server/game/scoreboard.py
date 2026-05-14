import csv
import os.path

from loguru import logger


class Scoreboard:
    def __init__(self, scoreboard_save_dir: str | None):
        self._scores: dict[str, Score] = dict()
        self._scores_save_path = scoreboard_save_dir
        self._game_index: int = 0

    def increment_kills(self, player_id: str):
        self._scores.setdefault(player_id, Score()).increment_kills()

    def increment_food_eaten(self, player_id: str):
        self._scores.setdefault(player_id, Score()).increment_food_eaten()

    def on_player_death(self, player_id: str, tick_index: int):
        self._scores.setdefault(player_id, Score()).set_num_ticks_lived(tick_index + 1)

    def save_and_reset(self):
        # save then clear
        self._save()
        self._scores.clear()
        self._game_index += 1

    def _save(self):
        if self._scores_save_path is None:
            logger.debug(f"Save path not specified, not saving...")
            return

        os.makedirs(self._scores_save_path, exist_ok=True)
        for player_id, score in self._scores.items():
            file_path = os.path.join(self._scores_save_path, f"{player_id}.csv")
            file_exists = os.path.exists(file_path)
            with open(os.path.join(self._scores_save_path, f"{player_id}.csv"), "a", newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["game_index", "food_eaten", "kills", "num_ticks_lived"]) # header
                writer.writerow([self._game_index, score.get_food_eaten(), score.get_kills(), score.get_num_ticks_lived()])



class Score:
    def __init__(self):
        self._food_eaten: int = 0
        self._kills: int = 0
        self._num_ticks_lived: int = 0

    def get_food_eaten(self) -> int:
        return self._food_eaten

    def get_kills(self) -> int:
        return self._kills

    def get_num_ticks_lived(self) -> int:
        return self._num_ticks_lived

    def increment_kills(self):
        self._kills += 1

    def increment_food_eaten(self):
        self._food_eaten += 1

    def set_num_ticks_lived(self, tick_index: int):
        self._num_ticks_lived = tick_index