import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.cube_env import CubeBalancingEnv

def select_model_interactive():
    models_dir = os.path.join("results", "models")
    all_dirs = [os.path.join(models_dir, d) for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    if not all_dirs:
        return None
        
    all_dirs.sort(key=os.path.getmtime, reverse=True)
    
    print("\n===============================")
    print("【バージョンの選択】")
    print("0: 最新の学習データ (latest)")
    for i, d in enumerate(all_dirs, 1):
        d_name = os.path.basename(d)
        print(f"{i}: {d_name}")
    print("===============================")
        
    chosen_dir = None
    while True:
        try:
            choice = input(f"\n見たいバージョンの番号を入力してください [0-{len(all_dirs)}]: ")
            if choice.strip() == "" or choice == "0":
                chosen_dir = all_dirs[0]
                break
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(all_dirs):
                chosen_dir = all_dirs[choice_idx]
                break
            else:
                print("※正しい番号を入力してください。")
        except ValueError:
            print("※数字を入力してください。")

    zips = glob.glob(os.path.join(chosen_dir, "*.zip"))
    if not zips:
        print(f"エラー: {os.path.basename(chosen_dir)} の中にモデルがありません。")
        return None
        
    zips.sort(key=os.path.getmtime, reverse=True)
    
    print("\n===============================")
    print(f"【{os.path.basename(chosen_dir)} 内のチェックポイント選択】")
    print("0: 最新のチェックポイント (デフォルト)")
    for i, file_path in enumerate(zips, 1):
        filename = os.path.basename(file_path)
        print(f"{i}: {filename}")
    print("===============================")
    
    while True:
        try:
            choice = input(f"\nロードするチェックポイントの番号を入力してください [0-{len(zips)}]: ")
            if choice.strip() == "" or choice == "0":
                return zips[0]
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(zips):
                return zips[choice_idx]
            else:
                print("※正しい番号を入力してください。")
        except ValueError:
            print("※数字を入力してください。")


def main():
    print("=== PyBullet 強化学習 (PPO) テストスクリプト ===")
    
    model_path = select_model_interactive()

    if not model_path or not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません。")
        print("先に 03_train_rl.py を実行してモデルを学習させてください。")
        return

    parent_dir = os.path.basename(os.path.dirname(model_path))
    filename = os.path.basename(model_path)
    print(f"\n【ロード完了】: {parent_dir} / {filename} を読み込みました。")

    # 再生スピードのメニューを削除（物理的な限界で機能しないため）
    env = CubeBalancingEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO.load(model_path)
    obs = vec_env.reset()
    
    dt = 1.0 / 240.0
    
    print("\nテストを開始します。")
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        # 常に等倍で描画
        time.sleep(dt)
        
        if done:
            print("モデルがバランスを崩してオペレーションが終了しました。リセットします。")
            obs = vec_env.reset()
            time.sleep(0.5)

if __name__ == "__main__":
    main()
