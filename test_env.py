import time
from envs.cube_env import CubeBalancingEnv

def main():
    print("=== PyBullet 物理シミュレーション・テスト ===")
    print("AIの代わりにランダムなモーター指令を与えて物理演算の挙動を確認します。")
    
    # 画面に描画するモード("human")で環境を立ち上げる
    env = CubeBalancingEnv(render_mode="human")
    
    # 環境の初期化（キューブが少し傾いた状態でスタートします）
    obs, info = env.reset()
    
    # 1000ステップ（約4秒分）だけ時間を進めてみる
    for i in range(1000):
        # AIの代わりに、ランダムなモーターのパワー（-1.0 〜 1.0）を決定
        # 実際にはここにAI（強化学習モデル）が入ります
        random_action = env.action_space.sample() 
        
        # 決定した行動を実行して、時間を1コマ進める
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        # 画面を見るために現実と同じスピード（1/240秒）だけ待つ
        time.sleep(1./240.)
        
        # もしキューブが45度以上倒れたら、リセットしてやり直す
        if terminated:
            print(f"ステップ {i} で倒れました！リセットします。")
            obs, info = env.reset()
            time.sleep(0.5) # リセット時に見やすいように一瞬止める
            
    env.close()
    print("テスト完了！")

if __name__ == "__main__":
    main()
