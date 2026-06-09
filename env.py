import random

import numpy as np

from config import (
    N,
    WIN_TILE,
    MAX_EPISODE_STEPS,
    ACTION_DIM,
    REWARD_EMPTY_WEIGHT,
    REWARD_MONOTONICITY_WEIGHT,
    REWARD_SMOOTHNESS_WEIGHT,
    REWARD_CORNER_WEIGHT,
    REWARD_EDGE_WEIGHT,
    REWARD_SNAKE_WEIGHT,
    REWARD_SECOND_MAX_WEIGHT,
    REWARD_MERGE_SCALE,
    REWARD_WIN,
    REWARD_LOSS,
    REWARD_CLIP,
    REWARD_NORM,
    REWARD_TILE_MILESTONES,
    ANCHOR_CORNER,
)


class Game2048Env:
    """2048 环境: 标准四方向滑动合并规则。"""

    def __init__(self, n=N, max_episode_steps=MAX_EPISODE_STEPS):
        self.n = n
        self.max_episode_steps = max_episode_steps
        self.board = np.zeros((n, n), dtype=np.int64)
        self.score = 0
        self.steps = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((self.n, self.n), dtype=np.int64)
        self.score = 0
        self.steps = 0
        self.spawn_tile()
        self.spawn_tile()
        return self._get_state()

    def spawn_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            r, c = random.choice(empty)
            self.board[r, c] = 4 if random.random() < 0.1 else 2

    @staticmethod
    def _merge_nonzero(cells):
        """合并一行/列中的非零格子, 返回 (merged_list, score_gain)。"""
        merged = []
        score = 0
        i = 0
        while i < len(cells):
            if i + 1 < len(cells) and cells[i] == cells[i + 1]:
                val = int(cells[i] * 2)
                merged.append(val)
                score += val
                i += 2
            else:
                merged.append(int(cells[i]))
                i += 1
        return merged, score

    @classmethod
    def _compress_line(cls, line):
        """将一条线向左压缩并合并。line 为长度 n 的 1d 数组。"""
        n = line.shape[0]
        cells = [int(x) for x in line if x != 0]
        merged, score = cls._merge_nonzero(cells)
        new_line = np.zeros(n, dtype=np.int64)
        for i, val in enumerate(merged):
            new_line[i] = val
        changed = not np.array_equal(line, new_line)
        return new_line, score, changed

    @classmethod
    def _move_board(cls, board, action):
        """
        按标准 2048 规则移动棋盘。
        action: 0=上 1=下 2=左 3=右
        返回 (new_board, score_gain) 或 (None, 0) 表示无效步。
        """
        n = board.shape[0]
        new_board = np.zeros_like(board)
        total_score = 0
        changed = False

        if action == 2:  # 左
            for r in range(n):
                row, score, ch = cls._compress_line(board[r])
                new_board[r] = row
                total_score += score
                changed = changed or ch
        elif action == 3:  # 右
            for r in range(n):
                row, score, ch = cls._compress_line(board[r][::-1])
                new_board[r] = row[::-1]
                total_score += score
                changed = changed or ch
        elif action == 0:  # 上
            for c in range(n):
                col, score, ch = cls._compress_line(board[:, c])
                new_board[:, c] = col
                total_score += score
                changed = changed or ch
        elif action == 1:  # 下
            for c in range(n):
                col, score, ch = cls._compress_line(board[:, c][::-1])
                new_board[:, c] = col[::-1]
                total_score += score
                changed = changed or ch
        else:
            return None, 0

        if not changed:
            return None, 0
        return new_board, total_score

    @classmethod
    def _preview_board(cls, board, action):
        result, _ = cls._move_board(board, action)
        return result

    def get_valid_actions(self):
        valid = []
        for action in range(ACTION_DIM):
            if self._preview_board(self.board, action) is not None:
                valid.append(action)
        return valid

    def get_action_mask(self):
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        for action in self.get_valid_actions():
            mask[action] = 1.0
        return mask

    def _anchor_position(self):
        if ANCHOR_CORNER == "bottom_left":
            return self.n - 1, 0
        if ANCHOR_CORNER == "bottom_right":
            return self.n - 1, self.n - 1
        if ANCHOR_CORNER == "top_left":
            return 0, 0
        return 0, self.n - 1

    def _board_heuristic(self, board):
        ar, ac = self._anchor_position()
        corner_val = int(board[ar, ac])
        corner_score = np.log2(corner_val) if corner_val > 0 else 0.0

        max_val = int(board.max())
        edge_score = 0.0
        if max_val > 0:
            edge_positions = []
            last = self.n - 1
            for idx in range(self.n):
                edge_positions.append((0, idx))
                edge_positions.append((last, idx))
                edge_positions.append((idx, 0))
                edge_positions.append((idx, last))
            seen = set()
            for r, c in edge_positions:
                if (r, c) in seen:
                    continue
                seen.add((r, c))
                if int(board[r, c]) == max_val:
                    edge_score = np.log2(max_val)
                    break

        return (
            REWARD_CORNER_WEIGHT * corner_score
            + REWARD_EDGE_WEIGHT * edge_score
        )

    def _milestone_bonus(self, old_max, new_max):
        bonus = 0.0
        for threshold, reward in REWARD_TILE_MILESTONES.items():
            if old_max < threshold <= new_max:
                bonus += reward
        return bonus

    def step(self, action):
        old_board = self.board.copy()
        old_heuristic = self._board_heuristic(old_board)

        new_board, merge_score = self._move_board(self.board, int(action))
        if new_board is None:
            return self._get_state(), REWARD_LOSS * 0.1, False, False

        post_move_board = new_board.copy()
        new_heuristic = self._board_heuristic(post_move_board)
        reward = (new_heuristic - old_heuristic) + merge_score * REWARD_MERGE_SCALE / 100.0

        old_max = int(old_board.max())
        new_max = int(post_move_board.max())
        reward += self._milestone_bonus(old_max, new_max)

        self.board = new_board
        self.score += int(merge_score)
        self.spawn_tile()
        self.steps += 1

        done = False
        if WIN_TILE in post_move_board:
            reward += REWARD_WIN
            done = True
        elif not self.get_valid_actions():
            reward += REWARD_LOSS
            done = True
        elif self.max_episode_steps > 0 and self.steps >= self.max_episode_steps:
            done = True

        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP)) / REWARD_NORM
        return self._get_state(), reward, done, True

    def _get_state(self):
        state = np.zeros_like(self.board, dtype=np.int64)
        nz = self.board > 0
        state[nz] = np.log2(self.board[nz]).astype(np.int64)
        return state

    def get_raw_board(self):
        return self.board.copy()
