import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import glob
import shutil
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from envs.cube_env import CubeBalancingEnv

class ShortCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(ShortCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            step_k = self.num_timesteps // 1000
            path = os.path.join(self.save_path, f"{step_k}k.zip")
            self.model.save(path)
            if self.verbose > 0:
                print(f"\n[SAVE] チェックポイントを保存しました: {step_k}k.zip")
        return True

def select_zip_in_dir(chosen_dir):
    zips = glob.glob(os.path.join(chosen_dir, "*.zip"))
    if not zips:
        return None
        
    zips.sort(key=os.path.getmtime, reverse=True)
    
    print("\n===============================")
    print(f"【{os.path.basename(chosen_dir)} 内のチェックポイント選択】")
    print("0: 最新のチェックポイント (デフォルト)")
    for i, file_path in enumerate(zips, 1):
        filename = os.path.basename(file_path)
        print(f"{i}: {filename} からロードする")
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


def select_training_mode():
    models_dir = "results"
    latest_dir = os.path.join(models_dir, "latest")
    
    archives = [os.path.join(models_dir, d) for d in os.listdir(models_dir) 
                if os.path.isdir(os.path.join(models_dir, d)) and d != "latest"]
    archives.sort(key=os.path.getmtime, reverse=True)
    
    print("\n===============================")
    print("【学習モードの選択】")
    print("0: [Initialize] 新規モデルを初期化して学習を開始する")
    
    has_latest = os.path.exists(latest_dir) and glob.glob(os.path.join(latest_dir, "*.zip"))
    options = []
    
    if has_latest:
        options.append(latest_dir)
        print("1: [Resume] 現在の latest からモデルをロードして学習を再開する")
        
    for d in archives:
        options.append(d)
        idx = len(options)
        d_name = os.path.basename(d)
        print(f"{idx}: [Resume] アーカイブ ({d_name}) からモデルをロードして学習を再開する")
    print("===============================")
    
    def archive_latest():
        if os.path.exists(latest_dir):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_dir = os.path.join(models_dir, timestamp)
            shutil.move(latest_dir, archive_dir)
            print(f"※既存の latest データを {timestamp} にアーカイブしました。")
            return timestamp
        return None
            
    while True:
        try:
            choice = input(f"\n番号を入力してEnterを押してください [0-{len(options)}]: ")
            if choice.strip() == "":
                choice_idx = 1 if has_latest else 0
            else:
                choice_idx = int(choice)
            
            # 0: Initialize
            if choice_idx == 0:
                archive_latest()
                os.makedirs(latest_dir, exist_ok=True)
                with open(os.path.join(latest_dir, "history.txt"), "w", encoding="utf-8") as f:
                    f.write("【Model Lineage】\n")
                    f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("このモデルは、新規に初期化されたベースモデルから学習を開始しています。\n")
                return None, latest_dir
                
            # 1以降: Resume
            elif 1 <= choice_idx <= len(options):
                chosen_dir = options[choice_idx - 1]
                
                chosen_zip = select_zip_in_dir(chosen_dir)
                if not chosen_zip:
                    print("※ そのディレクトリ内にモデルが見つかりません。")
                    continue
                
                zip_name = os.path.basename(chosen_zip)
                parent_name = os.path.basename(chosen_dir)
                is_latest = (parent_name == "latest")
                
                archived_timestamp = archive_latest()
                os.makedirs(latest_dir, exist_ok=True)
                
                if is_latest and archived_timestamp:
                    parent_name = archived_timestamp
                    chosen_zip = os.path.join(models_dir, archived_timestamp, zip_name)
                
                with open(os.path.join(latest_dir, "history.txt"), "w", encoding="utf-8") as f:
                    f.write("【Model Lineage】\n")
                    f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"このモデルは、アーカイブ [{parent_name}] の [{zip_name}] から\n")
                    f.write("既存の重みを引き継ぎ、新規ブランチとして学習を再開しています。\n")
                    
                return chosen_zip, latest_dir
            else:
                print("※正しい番号を入力してください。")
        except ValueError:
            print("※数字を入力してください。")


def main():
    print("=== PyBullet 強化学習 (PPO) 訓練スクリプト ===")
    
    models_dir = "results"
    os.makedirs(models_dir, exist_ok=True)
    
    resume_model_path, current_v_dir = select_training_mode()
    
    env = CubeBalancingEnv(render_mode="direct")
    env = Monitor(env)
    check_env(env)
    vec_env = DummyVecEnv([lambda: env])
    
    # 乱数シードの固定（再現性の確保）
    seed = 42
    set_random_seed(seed)
    vec_env.seed(seed)
    
    # 実機制御に向けたハイエンドPPOハイパーパラメータ
    custom_objects = {
        "learning_rate": 1e-4,  # 学習率を下げて破局的忘却（急激な記憶の上書き）を防止 (デフォルト 3e-4)
        "n_steps": 4096,        # 1回の学習に使う経験データを倍増し、より慎重に学習 (デフォルト 2048)
        "batch_size": 256,      # バッチサイズを大きくして勾配のブレを抑制 (デフォルト 64)
        "gamma": 0.995,         # より遠い未来の報酬を重視する（バランス維持に有効） (デフォルト 0.99)
        "ent_coef": 0.005,      # 探索を促し、1つの過激な操作への過学習を防ぐ (デフォルト 0.0)
    }
    
    if resume_model_path is None:
        print(f"\n【Initialize】新規モデルの学習を開始します。(保存先: latest)")
        # 新規学習時は脳みそ（ニューラルネットワーク）のサイズを拡張して複雑な制御に対応
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO("MlpPolicy", vec_env, verbose=1, 
                    tensorboard_log=os.path.join(current_v_dir, "logs"),
                    policy_kwargs=policy_kwargs,
                    **custom_objects)
        reset_timesteps = True
    else:
        print(f"\n【Resume】指定されたチェックポイントの重みを引き継いで学習を再開します。(保存先: latest)")
        # 古いモデルを引き継ぐ場合、脳のサイズは当時のままですが、学習率などの安全設定は上書き適用されます
        model = PPO.load(resume_model_path, env=vec_env, 
                         tensorboard_log=os.path.join(current_v_dir, "logs"),
                         custom_objects=custom_objects)
        reset_timesteps = False
    
    checkpoint_callback = ShortCheckpointCallback(
        save_freq=10000,
        save_path=current_v_dir
    )
    
    total_timesteps = 100000
    print(f"\n学習を開始します... (目標学習ステップ数: {total_timesteps})")
    
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=reset_timesteps)
    
    print(f"\n学習が完了しました。")

if __name__ == "__main__":
    main()
