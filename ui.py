import os
import sys

import numpy as np
import pygame

from agent import Agent
from config import (
    BEST_MODEL_PATH,
    CELL_SIZE,
    FPS,
    HEIGHT,
    MARGIN,
    MODEL_PATH,
    N,
    PANEL_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    WIDTH,
    WIN_TILE,
)
from env import Game2048Env

# 颜色配置
COLOR_BG = (187, 173, 160)
COLOR_TEXT = (119, 110, 101)
COLOR_TEXT_LIGHT = (249, 246, 242)

TILES_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_w: 0,
    pygame.K_s: 1,
    pygame.K_a: 2,
    pygame.K_d: 3,
}


class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font
        self.color = (143, 122, 102)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)
        text_surf = self.font.render(self.text, True, COLOR_TEXT_LIGHT)
        screen.blit(
            text_surf,
            (
                self.rect.centerx - text_surf.get_width() // 2,
                self.rect.centery - text_surf.get_height() // 2,
            ),
        )

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


def _load_agent_model(agent):
    """优先加载最优模型, 其次加载最新 checkpoint。"""
    for path in (BEST_MODEL_PATH, MODEL_PATH):
        if path and os.path.exists(path):
            agent.load_model(path)
            return path
    agent.load_model(MODEL_PATH)
    return None


def _tile_color(value):
    if value in TILES_COLORS:
        return TILES_COLORS[value]
    if value > 2048:
        return (60, 58, 50)
    return TILES_COLORS[0]


def _apply_action(env, action):
    """执行一步; 返回 (是否 Game Over, 是否有效步)。"""
    _, _, _, valid = env.step(int(action))
    if not valid:
        return False, False
    return len(env.get_valid_actions()) == 0, True


def play_ui():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2048 PPO Agent")

    try:
        font_large = pygame.font.SysFont("arial", 40, bold=True)
        font_medium = pygame.font.SysFont("arial", 25, bold=True)
        font_small = pygame.font.SysFont("arial", 18)
    except TypeError:
        font_large = pygame.font.Font(None, 40)
        font_medium = pygame.font.Font(None, 25)
        font_small = pygame.font.Font(None, 18)

    # UI 模式: max_episode_steps=0 表示不设步数上限
    env = Game2048Env(max_episode_steps=0)
    agent = Agent()
    loaded_path = _load_agent_model(agent)
    if loaded_path:
        print(f"UI loaded model: {loaded_path}")

    clock = pygame.time.Clock()

    is_playing = False
    game_over = False
    reached_2048 = False
    update_interval_ms = 500
    last_action_time = 0
    last_action_name = "-"
    last_action_valid = True

    panel_x = WIDTH + MARGIN
    btn_start = Button(panel_x, 50, 200, 40, "Start AI", font_medium)
    btn_stop = Button(panel_x, 100, 200, 40, "Pause AI", font_medium)
    btn_reset = Button(panel_x, 150, 200, 40, "Reset", font_medium)
    btn_speed_up = Button(panel_x, 200, 95, 40, "Speed +", font_small)
    btn_speed_down = Button(panel_x + 105, 200, 95, 40, "Speed -", font_small)

    running = True
    while running:
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if btn_start.is_clicked(event.pos):
                    if not game_over:
                        is_playing = True
                elif btn_stop.is_clicked(event.pos):
                    is_playing = False
                elif btn_reset.is_clicked(event.pos):
                    env.reset()
                    is_playing = False
                    game_over = False
                    reached_2048 = False
                    last_action_time = 0
                    last_action_name = "-"
                    last_action_valid = True
                elif btn_speed_up.is_clicked(event.pos):
                    update_interval_ms = max(50, update_interval_ms - 50)
                elif btn_speed_down.is_clicked(event.pos):
                    update_interval_ms += 50
            elif event.type == pygame.KEYDOWN and not game_over:
                action = KEY_TO_ACTION.get(event.key)
                if action is not None:
                    is_playing = False
                    game_over, valid = _apply_action(env, action)
                    last_action_name = ACTION_NAMES[action]
                    last_action_valid = valid
                    if np.max(env.board) >= WIN_TILE:
                        reached_2048 = True

        if is_playing and not game_over:
            if current_time - last_action_time > update_interval_ms:
                state = env._get_state()
                action, _, _ = agent.select_action(state, evaluate=True)

                # agent 已做动作掩码; 若仍无效则跳过本帧, 不做随机补救
                if action not in env.get_valid_actions():
                    valid_actions = env.get_valid_actions()
                    if not valid_actions:
                        game_over = True
                        is_playing = False
                    last_action_time = current_time
                    continue

                ended, valid = _apply_action(env, action)
                last_action_name = ACTION_NAMES[action]
                last_action_valid = valid
                if ended:
                    game_over = True
                    is_playing = False
                if np.max(env.board) >= WIN_TILE:
                    reached_2048 = True

                last_action_time = current_time

        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, (250, 248, 239), (WIDTH, 0, PANEL_WIDTH, SCREEN_HEIGHT))

        for r in range(N):
            for c in range(N):
                val = int(env.board[r, c])
                col = _tile_color(val)
                rect = pygame.Rect(
                    MARGIN + c * (CELL_SIZE + MARGIN),
                    MARGIN + r * (CELL_SIZE + MARGIN),
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(screen, col, rect, border_radius=5)
                if val > 0:
                    text_col = COLOR_TEXT if val <= 4 else COLOR_TEXT_LIGHT
                    txt_surf = font_large.render(str(val), True, text_col)
                    screen.blit(
                        txt_surf,
                        (
                            rect.centerx - txt_surf.get_width() // 2,
                            rect.centery - txt_surf.get_height() // 2,
                        ),
                    )

        btn_start.draw(screen)
        btn_stop.draw(screen)
        btn_reset.draw(screen)
        btn_speed_up.draw(screen)
        btn_speed_down.draw(screen)

        if game_over:
            status = "GAME OVER"
        elif is_playing:
            status = "Playing (AI)"
        elif reached_2048:
            status = "Reached 2048!"
        else:
            status = "Paused (Arrow/WASD)"

        info_texts = [
            f"Score: {env.score}",
            f"Steps: {env.steps}",
            f"Max Tile: {int(np.max(env.board))}",
            f"Last Move: {last_action_name}" + ("" if last_action_valid else " (invalid)"),
            f"Speed: {update_interval_ms} ms",
            f"Status: {status}",
        ]
        if loaded_path:
            info_texts.append(f"Model: {loaded_path}")

        y_offset = 270
        for info in info_texts:
            surf = font_medium.render(info, True, COLOR_TEXT)
            screen.blit(surf, (panel_x, y_offset))
            y_offset += 36

        hint = font_small.render("Keys: Up Down Left Right / WASD", True, COLOR_TEXT)
        screen.blit(hint, (panel_x, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    play_ui()
