import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from envs.cube_env import CubeBalancingEnv

def main():
    print("=== PyBullet 物理シミュレーション・テスト ===")
    print("AIの代わりにランダムなモーター指令を与えて物理演算の挙動を確認します。")
    
    # 画面に描画するモード("human")で環境を立ち上げる
    env = CubeBalancingEnv(render_mode="human")
    
    # 環境の初期化（キューブが少し傾いた状態でスタートします）
    obs, info = env.reset()
    
    # 無限ループに変更（画面を閉じるまで永遠に見れます）
    while True:
        # AIの代わりに、ランダムなモーターのパワー（-1.0 〜 1.0）を決定
        # 実際にはここにAI（強化学習モデル）が入ります
        random_action = env.action_space.sample() 
        
        # 決定した行動を実行して、時間を1コマ進める
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        # 画面を見るために現実と同じスピード（1/240秒）だけ待つ
        time.sleep(1.0 / 240.0)
        
        # もしキューブが45度以上倒れたら、リセットしてやり直す
        if False:
            print("倒れました！リセットします。")
            obs, info = env.reset()
            time.sleep(0.5)

if __name__ == "__main__":
    main()
