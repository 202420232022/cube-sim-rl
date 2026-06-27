import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.cube_env import CubeBalancingEnv

def main():
    print("=== PyBullet 強化学習 (PPO) テストスクリプト ===")
    
    model_path = os.path.join("results", "models", "ppo_cube.zip")
    if not os.path.exists(model_path):
        print(f"エラー: モデルファイルが見つかりません: {model_path}")
        print("先に 03_train_rl.py を実行してAIを学習させてください。")
        return

    # 画面描画あり(HUMANモード)で環境を作成
    env = CubeBalancingEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])
    
    # 学習済みのモデル（脳）をロード
    print("モデルを読み込み中...")
    model = PPO.load(model_path)
    
    # AIによる推論と操作のループ
    obs = vec_env.reset()
    
    # 物理演算の1ステップの時間 (1/240秒)
    dt = 1.0 / 240.0
    
    print("テストを開始します。")
    while True:
        # AIが現在の状態（obs）を見て、最適な行動（action）を予測
        # deterministic=True にすると、確率的なブレをなくし、AIの「最善手」のみを実行します
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
