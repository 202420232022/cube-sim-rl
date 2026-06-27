import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.cube_env import CubeBalancingEnv

def main():
    print("=== PyBullet 強化学習 (PPO) 訓練スクリプト ===")
    
    # モデル保存用のディレクトリを作成
    models_dir = os.path.join("results", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 画面描画なし(DIRECTモード)で環境を作成し、計算速度を最大化する
    env = CubeBalancingEnv(render_mode="direct")
    
    # 環境をMonitorでラップすることで、「何回目の試行か」などの詳しい情報がログに出るようになります
    env = Monitor(env)
    
    # (オプション) Gym環境の構造が正しいかチェック
    check_env(env)
    
    # Stable Baselines3が扱いやすいように環境をラップする
    vec_env = DummyVecEnv([lambda: env])
    
    # PPOモデルの初期化（頭脳の作成）
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./results/tensorboard/")
    
    # 10000ステップ（約15秒）ごとに「その時点の脳みそ」を保存する設定
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=models_dir,
        name_prefix="ppo_cube_step"
    )
    
    # 学習の実行 (例: 10万ステップ)
    total_timesteps = 100000
    print(f"\n学習を開始します... (目標: {total_timesteps} ステップ)")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    # 学習したモデル（脳）を保存
    save_path = os.path.join(models_dir, "ppo_cube")
    model.save(save_path)
    print(f"\n学習が完了しました！ モデルを保存しました: {save_path}.zip")

if __name__ == "__main__":
    main()
