import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import os

class CubeBalancingEnv(gym.Env):
    """
    1辺倒立キューブ用強化学習環境。
    AIに対して「現在の角度」「角速度」を教え、「モーターのトルク」を受け取り、
    「報酬」を返すためのインターフェースです。
    """
    
    def __init__(self, render_mode=None):
        super(CubeBalancingEnv, self).__init__()
        
        self.render_mode = render_mode
        
        # --- 1. AIができる「行動 (Action)」の定義 ---
        # モーターへの入力（例：-1.0〜1.0 のトルク指示値）
        # 【変更ポイント】もしモーターをもっと細かく制御させたい場合はここを変えます
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # --- 2. AIが見える「状態 (Observation)」の定義 ---
        # AIが知ることができるセンサー情報（IMUの代わり）
        # [現在の傾き角度(rad), 現在の角速度(rad/s), フライホイールの回転速度(rad/s)]
        high = np.array([np.pi, 100.0, 500.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        
        # --- PyBullet (物理エンジン) の起動 ---
        if self.render_mode == "human":
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 内部変数
        self.robot_id = None
        self.motor_joint_index = 1 # URDFの2番目のジョイントがモーター
        
        # 【モーターパワーの最大値】URDFの数値をオーバーライドできます
        # カリキュラムフェーズ1（初期学習）のため、本来の限界値(0.03)の10倍に設定しています
        self.max_motor_torque = 0.3
        
        # 【エピソードの最大ステップ数】
        self.current_step = 0
        self.max_steps = 1000
        
        # 【ドメインランダマイゼーション (Sim-to-Real用)】
        # カリキュラムフェーズ1（初期学習）のため、ノイズを無効化しています
        self.noise_std_angle = 0.0      # 角度のノイズ
        self.noise_std_gyro = 0.0       # 角速度のノイズ
        self.noise_std_motor = 0.0      # モーター回転数のノイズ

    def reset(self, seed=None, options=None):
        """
        環境の初期化。AIが失敗してやり直すたびに呼ばれます。
        """
        super().reset(seed=seed)
        p.resetSimulation()
        # 床を読み込む（衝突マージンによるバグを防ぐため、少し下に下げます）
        p.loadURDF("plane.urdf", basePosition=[0, 0, -0.05])
        
        # キューブの物理モデル(URDF)を読み込む
        urdf_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'cube.urdf')
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
        
        p.setGravity(0, 0, -9.81)
        
        # 【初期状態の変更】
        # カリキュラムフェーズ1（初期学習）のため、ほぼ直立の状態からスタートさせます
        initial_angle = self.np_random.uniform(low=-0.01, high=0.01) # -0.5度 〜 +0.5度
        p.resetJointState(self.robot_id, 0, targetValue=initial_angle) # 床のヒンジ(joint 0)を傾ける
        
        # PyBulletはデフォルトで全ての関節に「位置を保持するモーター」がオンになっています。
        # 床のヒンジ（joint 0）が勝手に固定されないよう、保持モーターをオフ(force=0)にして重力で自然に倒れるようにします。
        # ※必ず resetJointState の後に呼ぶ必要があります。
        p.setJointMotorControl2(self.robot_id, 0, controlMode=p.VELOCITY_CONTROL, force=0)
        
        # モーター（joint 1）の摩擦抵抗もオフにして、トルク制御モードにする準備
        p.setJointMotorControl2(self.robot_id, self.motor_joint_index, controlMode=p.VELOCITY_CONTROL, force=0)
        
        # キューブを物理エンジン上で「スリープ（静止状態）」から強制的に「ウェイクアップ（計算開始）」させます
        p.changeDynamics(self.robot_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
        p.changeDynamics(self.robot_id, 0, activationState=p.ACTIVATION_STATE_WAKE_UP)
        
        self.current_step = 0
        
        return self._get_obs(), {}

    def step(self, action):
        """
        AIが「行動(モーターを回す)」を選択したときに呼ばれ、時間を1コマ進めます
        """
        # 1. 床のヒンジ(joint 0)が勝手に固定されないよう念のため毎ステップ保持力を0に設定
        p.setJointMotorControl2(
            self.robot_id,
            0,
            controlMode=p.TORQUE_CONTROL,
            force=0
        )
        
        # 2. AIからの行動（-1.0〜1.0）を実際のトルクに変換してモーター(joint 1)に適用
        torque = float(np.clip(action[0], -1.0, 1.0)) * self.max_motor_torque
        p.setJointMotorControl2(
            self.robot_id, 
            self.motor_joint_index, 
            controlMode=p.TORQUE_CONTROL, 
            force=torque
        )
        
        # 3. アクションリピート (4コマ分同じトルクをかけ続ける = 60Hz制御)
        for _ in range(4):
            p.stepSimulation()
            
        self.current_step += 1
        
        # 3. 現在の状況を取得
        obs = self._get_obs()
        current_angle = obs[0]
        
        # 4. 報酬の計算（AIへの評価）と、ゲームオーバー判定
        reward = self._compute_reward(obs)
        
        # 【ゲームオーバーのルール】
        # もし45度(約0.78rad)以上傾いたら倒れたとみなして終了(Terminated)
        terminated = bool(abs(current_angle) > (math.pi / 4.0))
        # 1000ステップ（約16.6秒）耐えきったら時間切れクリア(Truncated)
        truncated = bool(self.current_step >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        現在のセンサー値(角度、角速度など)を取得してAIに渡す関数
        """
        # ジョイント0（床とのヒンジ＝キューブの傾き）の状態を取得
        cube_state = p.getJointState(self.robot_id, 0)
        true_cube_angle = cube_state[0]
        true_cube_velocity = cube_state[1]
        
        # ジョイント1（モーター）の状態を取得
        motor_state = p.getJointState(self.robot_id, 1)
        true_wheel_velocity = motor_state[1]
        
        # 【Sim-to-Real】現実のセンサーのブレ（ノイズ）を意図的に付加する
        noisy_cube_angle = true_cube_angle + self.np_random.normal(0, self.noise_std_angle)
        noisy_cube_velocity = true_cube_velocity + self.np_random.normal(0, self.noise_std_gyro)
        noisy_wheel_velocity = true_wheel_velocity + self.np_random.normal(0, self.noise_std_motor)
        
        return np.array([noisy_cube_angle, noisy_cube_velocity, noisy_wheel_velocity], dtype=np.float32)

    def _compute_reward(self, obs):
        """
        報酬関数の定義（Reward Shaping）
        モデルが倒立状態を安定して維持するように、各状態パラメータに対する報酬・ペナルティを計算します。
        """
        cube_angle = obs[0]
        cube_velocity = obs[1]
        wheel_velocity = obs[2]
        
        # 生存報酬 (Survival Reward)
        # ※方策崩壊（早期終了方策への陥り）を防止するため、ベースとなる生存報酬を高めに設定しています。
        reward = 10.0 
        
        # 角度ペナルティ (Angle Penalty)
        # ※微小な姿勢のブレに対する勾配を確保するため、二次関数ではなく絶対値を利用しています。
        angle_penalty = abs(cube_angle) * 10.0
        
        # 角速度ペナルティ (Velocity Penalty)
        # ※姿勢の安定化（振動の抑制）を促します。
        vel_penalty = (cube_velocity ** 2) * 0.1
        
        # 制御入力ペナルティ (Control Effort Penalty)
        # ※モータの過剰な回転を抑制します。係数を大きくし過ぎると早期終了方策を引き起こすため微小値に設定しています。
        wheel_penalty = (wheel_velocity ** 2) * 0.00001
        
        # 各ペナルティ項の減算
        total_reward = reward - angle_penalty - vel_penalty - wheel_penalty
        
        # 終了ペナルティ (Terminal Penalty)
        # ※方策崩壊（Policy Collapse）を防ぐため、エピソード終了条件（45度以上の傾き）を満たした場合に強力な負の報酬を与えます。
        if abs(cube_angle) > (math.pi / 4.0):
            total_reward -= 500.0
            
        return float(total_reward)

    def render(self):
        # 今回はPyBulletのGUIが自動で描画するため何もしなくてOKです
        pass

    def close(self):
        p.disconnect(self.physicsClient)
