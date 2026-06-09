"""2048 项目主入口: 集成所有方法"""

import sys
import os


def show_menu():
    print("\n" + "=" * 54)
    print("          2048 —— 人工智能方法综合实验")
    print("=" * 54)
    print("  1. 训练 PPO (强化学习, Part 5)")
    print("  2. 启动可视化界面 (UI)")
    print("  3. 评估启发式搜索算法 (Part 2)")
    print("  4. 遗传算法优化启发式参数 (Part 3)")
    print("  5. 监督学习训练 (Part 4)")
    print("  6. 快速验证 (50 updates PPO)")
    print("  0. 退出")
    print("=" * 54)


def main():
    while True:
        show_menu()
        choice = input("请输入对应数字: ").strip()

        if choice == "1":
            print("\n启动 PPO 训练 (Part 5)...")
            import train
            train.train()
            break

        elif choice == "2":
            print("\n启动可视化界面...")
            import ui
            ui.play_ui()
            break

        elif choice == "3":
            print("\n评估启发式搜索算法 (Part 2)...")
            import heuristic_search
            heuristic_search.run_heuristic_eval()
            input("\n按 Enter 返回菜单...")

        elif choice == "4":
            print("\n运行遗传算法优化 (Part 3)...")
            import evo_optimize
            evo_optimize.run_evo_optimization()
            input("\n按 Enter 返回菜单...")

        elif choice == "5":
            print("\n运行监督学习训练 (Part 4)...")
            import supervised_train
            supervised_train.run_supervised_training()
            input("\n按 Enter 返回菜单...")

        elif choice == "6":
            print("\n快速验证 (50 updates PPO)...")
            import run_short
            run_short.train()
            break

        elif choice == "0":
            print("程序退出。")
            sys.exit(0)

        else:
            print("输入无效，请重新选择。")


if __name__ == "__main__":
    main()
