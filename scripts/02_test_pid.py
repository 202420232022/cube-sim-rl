import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import math
from envs.cube_env import CubeBalancingEnv

def main():
    print("=== PyBullet 古典制御(PID) テスト ===")
    print("AIの代わりに、ラグランジアンから導出した運動方程式に基づくPID制御でバランスを取ります。")
    
    # 画面に描画するモード("human")で環境を立ち上げる
    env = CubeBalancingEnv(render_mode="human")
    
    # 環境の初期化（キューブが少しランダムに傾いた状態でスタートします）
    obs, info = env.reset()
    
    # 【PIDゲインの設定】
    # この数値をいじると、立ち上がりの速さや振動の大きさが変わります
    Kp = 0.45  # 比例ゲイン（傾きに比例してどれだけ強く戻すか）
    Kd = 0.3  # 微分ゲイン（どれくらいブレーキをかけて揺れを止めるか）
    Ki = 0.0  # 積分ゲイン（今回は不要）
    
    integral = 0.0
    dt = 1.0 / 240.0 # PyBulletの物理演算の1コマの時間
    
    import random
    
    # 無限ループで制御し続けます
    while True:
        # 現在の状態を取得
        angle = obs[0]
        velocity = obs[1]
        
        # --- 【現実世界を再現：センサーノイズの注入】 ---
        # 現実のIMUセンサーは振動や電気的ノイズで常に値がブレます。
        # ここでは、標準偏差 0.02ラジアン（約1度）のランダムなノイズを意図的に混ぜます。
        angle_noisy = angle + random.gauss(0, 0.02)
        velocity_noisy = velocity + random.gauss(0, 0.1) # 角速度はよりノイズが乗りやすい
        
        # 1. エラー（目標角度0度との差）を計算
        error = 0.0 - angle_noisy
        integral += error * dt
        
        # 2. 微分項の計算（Derivative Kick対策として速度の反転を利用）
        derivative = -velocity_noisy 
        
        # 3. PIDの計算式で必要な「キューブを戻すためのトルク」を算出
        torque = (Kp * error) + (Ki * integral) + (Kd * derivative)
        
        # 4. モーターへの入力値（-1.0 〜 1.0）に変換
        # 物理の反作用（作用・反作用の法則）により、モーターを右に回すとキューブは左に倒れようとします。
        # そのため、計算したトルクと「逆向き」にモーターを回す必要があります。
        max_torque = env.unwrapped.max_motor_torque
        action = -torque / max_torque
        
        # 決定した行動を実行して、時間を1コマ進める
        obs, reward, terminated, truncated, info = env.step([action])
        
        # 画面を見るために現実と同じスピード（1/240秒）だけ待つ
        time.sleep(dt)
        
        # もし45度以上倒れてしまったら（ゲイン調整ミスなどで）、リセットしてやり直す
        if terminated:
            print("バランスを崩して倒れました！リセットします。")
            obs, info = env.reset()
            integral = 0.0
            time.sleep(0.5)

if __name__ == "__main__":
    main()
