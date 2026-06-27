import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.cube_env import CubeBalancingEnv

def main():
    print("=== PyBullet 強化学習 (PPO) 訓練スクリプト ===")
    
    # モデル保存用のディレクトリを作成
    models_dir = os.path.join("results", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 画面を描画しながら(HUMANモード)学習を行い、上達の過程を見れるようにします！
    env = CubeBalancingEnv(render_mode="human")
    
    # (オプション) Gym環境の構造が正しいかチェック
    check_env(env)
    
    # Stable Baselines3が扱いやすいように環境をラップする
    vec_env = DummyVecEnv([lambda: env])
    
    # PPOモデルの初期化（頭脳の作成）
    # MlpPolicy: 状態（角度や角速度）から行動（トルク）を出力する多層パーセプトロン
    # verbose=1: 学習の進捗状況をターミナルに表示する
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./results/tensorboard/")
    
    # 学習の実行 (例: 10万ステップ)
    # 1ステップ = 1/240秒なので、10万ステップは約416秒（約7分）分の経験に相当します。
    # ※パソコンの性能にもよりますが、現実時間で数分で終わります。
    total_timesteps = 100000
    print(f"\n学習を開始します... (目標: {total_timesteps} ステップ)")
    model.learn(total_timesteps=total_timesteps)
    
    # 学習したモデル（脳）を保存
    save_path = os.path.join(models_dir, "ppo_cube")
    model.save(save_path)
    print(f"\n学習が完了しました！ モデルを保存しました: {save_path}.zip")

if __name__ == "__main__":
    main()
