"""2048 项目主入口: 集成所有方法"""

import sys


def show_menu():
    print("\n" + "=" * 60)
    print("          2048 —— 人工智能方法综合实验")
    print("=" * 60)
    print("  1. UI 调用 (启发式搜索 / 监督学习 / PPO)")
    print("  2. PPO 训练 (断点续训)")
    print("  3. 监督学习 — 启发式搜索数据")
    print("  4. 监督学习 — PPO 冠军模型数据")
    print("  5. 启发式算法 + 遗传算法参数优化")
    print("  6. 退出")
    print("=" * 60)


def main():
    while True:
        show_menu()
        choice = input("请输入对应数字: ").strip()

        if choice == "1":
            print("\n启动可视化界面 (可切换启发式 / 监督学习 / PPO 模型)...")
            import ui
            ui.play_ui()
            break

        elif choice == "2":
            print("\n启动 PPO 训练...")
            import train
            train.train()
            break

        elif choice == "3":
            print("\n监督学习 — 启发式搜索数据训练...")
            import supervised_train
            supervised_train.run_supervised_training()
            input("\n按 Enter 返回菜单...")

        elif choice == "4":
            print("\n监督学习 — PPO 冠军模型数据训练...")
            import supervised_train
            supervised_train.run_supervised_training_from_championship()
            input("\n按 Enter 返回菜单...")

        elif choice == "5":
            print("\n运行启发式评估 + 遗传算法参数优化...")
            import cross
            cross.run_heuristic_eval()
            print("\n接下来运行遗传算法优化...")
            cross.run_ga_optimization()
            input("\n按 Enter 返回菜单...")

        elif choice == "6":
            print("程序退出。")
            sys.exit(0)

        else:
            print("输入无效，请重新选择。")


if __name__ == "__main__":
    main()
