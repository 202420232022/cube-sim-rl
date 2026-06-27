import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import glob
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.cube_env import CubeBalancingEnv

def select_model_interactive():
    """
    保存されているモデル一覧を表示し、ユーザーに選ばせる機能
    """
    models_dir = os.path.join("results", "models")
    zip_files = glob.glob(os.path.join(models_dir, "*.zip"))
    
    if not zip_files:
        return None
        
    # 更新日時が新しい順（最新が上）に並び替える
    zip_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\n===============================")
    print("保存されているモデル一覧:")
    print("0: 最新のモデル (デフォルト)")
    for i, file_path in enumerate(zip_files, 1):
        filename = os.path.basename(file_path)
        print(f"{i}: {filename}")
    print("===============================")
        
    while True:
        try:
            choice = input(f"\n見たいモデルの番号を数字で入力してEnterを押してください [0-{len(zip_files)}]: ")
            # エンターキーだけ押された場合や0の場合は最新を選ぶ
            if choice.strip() == "" or choice == "0":
                return zip_files[0]
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(zip_files):
                return zip_files[choice_idx]
            else:
                print("※リストにある正しい番号を入力してください。")
        except ValueError:
            print("※数字を入力してください。")


def main():
    parser = argparse.ArgumentParser()
    # --step は残しつつ、何も指定されなかったら対話メニューを出す
    parser.add_argument('--step', type=int, help='ロードするチェックポイントのステップ数 (例: 10000)')
    args = parser.parse_args()

    print("=== PyBullet 強化学習 (PPO) テストスクリプト ===")
    
    if args.step:
        model_path = os.path.join("results", "models", f"ppo_cube_step_{args.step}_steps.zip")
    else:
        # 何も指定されていない場合はメニューを表示して選ばせる
        model_path = select_model_interactive()

    if not model_path or not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        print("先に 03_train_rl.py を実行してAIを学習させてください。")
        return

    print(f"\n【ロード完了】: {os.path.basename(model_path)} を読み込みました！")

    # 画面描画あり(HUMANモード)で環境を作成
    env = CubeBalancingEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])
    
    # 学習済みのモデル（脳）をロード
    model = PPO.load(model_path)
    
    # AIによる推論と操作のループ
    obs = vec_env.reset()
    
    # 物理演算の1ステップの時間 (1/240秒)
    dt = 1.0 / 240.0
    
    print("テストを開始します。")
    while True:
        # AIが現在の状態（obs）を見て、最適な行動（action）を予測
        action, _states = model.predict(obs, deterministic=True)
        
        # 環境に行動を適用して1コマ進める
        obs, reward, done, info = vec_env.step(action)
        
        # 画面描画を見るため、現実のスピードに合わせる
        time.sleep(dt)
        
        if done:
            print("AIがバランスを崩してゲームオーバーになりました！リセットします。")
            obs = vec_env.reset()
            time.sleep(0.5)

if __name__ == "__main__":
    main()
