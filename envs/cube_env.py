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
        self.max_motor_torque = 0.5 

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
        # 最初から直立だとAIが学習しないため、わざと少しランダムに傾けた状態からスタートさせます
        initial_angle = self.np_random.uniform(low=-0.2, high=0.2) # -11度 〜 +11度くらい
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
        
        # 2. 物理シミュレーションを1ステップ進める (1/240秒)
        p.stepSimulation()
        
        # 3. 現在の状況を取得
        obs = self._get_obs()
        current_angle = obs[0]
        
        # 4. 報酬の計算（AIへの評価）と、ゲームオーバー判定
        reward = self._compute_reward(obs)
        
        # 【ゲームオーバーのルール】
        # もし45度(約0.78rad)以上傾いたら倒れたとみなして終了(Terminated)
        terminated = bool(abs(current_angle) > (math.pi / 4.0))
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        """
        現在のセンサー値(角度、角速度など)を取得してAIに渡す関数
        """
        # ジョイント0（床とのヒンジ＝キューブの傾き）の状態を取得
        cube_state = p.getJointState(self.robot_id, 0)
        cube_angle = cube_state[0]
        cube_velocity = cube_state[1]
        
        # ジョイント1（モーター）の状態を取得
        motor_state = p.getJointState(self.robot_id, 1)
        wheel_velocity = motor_state[1]
        
        return np.array([cube_angle, cube_velocity, wheel_velocity], dtype=np.float32)

    def _compute_reward(self, obs):
        """
        【最も重要なAIのしつけ(報酬設計)】
        ここで「どういう状態が素晴らしいか」を定義します。
        """
        cube_angle = obs[0]
        cube_velocity = obs[1]
        wheel_velocity = obs[2]
        
        # ベース報酬：倒れていなければ毎ステップもらえる「生存ポイント」
        reward = 2.0 
        
        # 減点1：直立（0度）から傾いているほど減点
        angle_penalty = (cube_angle ** 2) * 10.0
        
        # 減点2：無駄にキューブがグラグラ揺れていたら減点
        vel_penalty = (cube_velocity ** 2) * 0.1
        
        # 減点3：モーター（フライホイール）が暴走して超高速回転していたら減点
        # ※以前はここが厳しすぎて、モーターを回すくらいなら自ら倒れてゲームオーバーになる（自殺する）というバグが起きていました！
        wheel_penalty = (wheel_velocity ** 2) * 0.00001
        
        # 全部の減点を引いたものが今回のスコア
        total_reward = reward - angle_penalty - vel_penalty - wheel_penalty
        
        # 【死への恐怖を教える】
        # もし倒れてしまったら、とてつもない罰を与えて「絶対に倒れてはいけない」と教え込みます
        if abs(cube_angle) > (math.pi / 4.0):
            total_reward -= 50.0
            
        return float(total_reward)

    def render(self):
        # 今回はPyBulletのGUIが自動で描画するため何もしなくてOKです
        pass

    def close(self):
        p.disconnect(self.physicsClient)
