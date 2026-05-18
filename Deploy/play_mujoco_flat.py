import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import os
import time
import glfw

class Cfg:
    # 路径配置 
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    ROBOT_ROOT = os.path.join(PROJECT_ROOT, "Go2_model")
    XML_PATH = os.path.join(ROBOT_ROOT, "xml/scene.xml")
    MESHES_DIR = os.path.join(ROBOT_ROOT, "xml/assets")
    POLICY_PATH = os.path.join(PROJECT_ROOT, "policy/policy.onnx")

    # 物理控制参数(与velocity_env_cfg.py中保持一致) 
    sim_duration = 9999.0   # 仿真持续时间
    sim_dt = 0.005  # 仿真频率
    decimation = 4 # 策略频率

    # 默认关节角度 
    default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 
                                0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.double)
    
    # PD控制参数
    kp = np.array([60.0] * 12, dtype=np.double)
    kd = np.array([2.0] * 12, dtype=np.double)
    tau_limit = 20.0

    # 观测值归一化参数
    class ObsScales:
        ang_vel = 0.25
        lin_vel = 2.0
        dof_pos = 1.0
        dof_vel = 0.05
    clip_obs = 5.0

    # 操控灵敏度 
    vel_lin_step = 0.1  # 线速度增量
    vel_ang_step = 0.1   # 角速度增量
    vel_decay = 0.4     # 速度自然衰减

# ========================= 工具函数 (Utils) =========================
"""
data shape:
    0:12 joint_pos
    12:24 joint_vel
    24:36 joint_actuatorfrc
    36:40 framequat [w, x ,y ,z]
    40:43 ang_vel   [x ,y ,z]
    43:46 lin_acc   [x ,y ,z]
    46:49 base_pos  [x ,y ,z]
    49:52 base_lin_vel  [x ,y ,z]  
"""

"""
sim_obs_data shape:
    base_ang_vel 3
    projected_gravity 3
    velocity_commands 3
    joint_pos 12
    joint_vel 12
    actions_smooth 12
"""

def remap_mujoco_to_net(arr):
    """
    将 MuJoCo 默认的 [FL, FR, RL, RR] 12维数组
    重排为网络和电机统一的 [FR, FL, RR, RL] 顺序
    """
    # 索引映射法则
    idx = [3, 4, 5,   # FR (原 3,4,5)
           0, 1, 2,   # FL (原 0,1,2)
           9, 10, 11, # RR (原 9,10,11)
           6, 7, 8]   # RL (原 6,7,8)
    return arr[idx]

def quat_rotate_inverse(q, v):
    """四元数反向旋转"""
    q_w, q_vec = q[-1], q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c
def get_obs(data):
    """提取机器人状态"""
    q = data.qpos.astype(np.double)[-12:]
    dq = data.qvel.astype(np.double)[-12:]
    q = remap_mujoco_to_net(q)
    dq = remap_mujoco_to_net(dq)
    quat = data.sensor('imu_quat').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('imu_gyro').data.astype(np.double)
    return q, dq, quat, omega

class ControllerState:
    cmd_vel = np.zeros(3)
def key_callback(keycode):
    """官方 viewer 触发的按键回调函数"""
    if keycode == glfw.KEY_UP:
        ControllerState.cmd_vel[0] += Cfg.vel_lin_step
    elif keycode == glfw.KEY_DOWN:
        ControllerState.cmd_vel[0] -= Cfg.vel_lin_step
    elif keycode == glfw.KEY_LEFT:
        ControllerState.cmd_vel[2] += Cfg.vel_ang_step
    elif keycode == glfw.KEY_RIGHT:
        ControllerState.cmd_vel[2] -= Cfg.vel_ang_step
    elif keycode == glfw.KEY_ENTER:
        ControllerState.cmd_vel[:] = 0.0  # 急停
def update_cmd_vel():

    cmd = ControllerState.cmd_vel
    cmd[0:2] = np.clip(cmd[0:2], -1.5, 1.5)
    cmd[2]   = np.clip(cmd[2], -1.0, 1.0)
    
    if np.linalg.norm(cmd) < 0.01: 
        cmd[:] = 0.0
        
    return cmd.copy()


if __name__ == '__main__':
    print(f"✅ 加载策略: {os.path.basename(Cfg.POLICY_PATH)}")

    # 初始化推理引擎
    policy = ort.InferenceSession(Cfg.POLICY_PATH, providers=['CPUExecutionProvider'])
    input_name = policy.get_inputs()[0].name
    output_name = policy.get_outputs()[0].name

    # 初始化环境
    m = mujoco.MjModel.from_xml_path(Cfg.XML_PATH)
    m.opt.timestep = Cfg.sim_dt
    d = mujoco.MjData(m)

    # 运行时变量
    cmd_vel = np.zeros(3)
    last_action = np.zeros(12, dtype=np.float32)
    target_q = Cfg.default_dof_pos.copy()

    print("\n" + "="*50)
    print("🤖 Unitree Go2 仿真就绪 (Stable Polling Ver.)")
    print("🎮 控制: [↑/↓] 前后 | [←/→] 转向 | [Enter] 急停")
    print("ℹ️  提示: 请确保点击黑色仿真窗口以获取焦点")
    print("="*50 + "\n")

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, key_callback=key_callback) as viewer:
        start_time = time.time()
        steps = int(Cfg.sim_duration / Cfg.sim_dt)

        for i in range(steps):
            if not viewer.is_running():
                break

            step_start = time.time()
            cmd_vel = update_cmd_vel()

            # --- 策略推理 (50Hz) ---
            if i % Cfg.decimation == 0:
                q, dq, quat, omega = get_obs(d)
                proj_gravity = quat_rotate_inverse(quat, np.array([0., 0., -1.]))
            
                obs_list = [
                    omega * Cfg.ObsScales.ang_vel,
                    proj_gravity,
                    cmd_vel * [Cfg.ObsScales.lin_vel, Cfg.ObsScales.lin_vel, Cfg.ObsScales.ang_vel],
                    (q - Cfg.default_dof_pos) * Cfg.ObsScales.dof_pos,
                    dq * Cfg.ObsScales.dof_vel,
                    last_action
                    ]
                obs = np.concatenate(obs_list).astype(np.float32).reshape(1, -1)
                obs = np.clip(obs, -Cfg.clip_obs, Cfg.clip_obs)

                # 推理
                raw_action = policy.run([output_name], {input_name: obs})[0][0]
                raw_action = np.clip(raw_action, -10, 10)
                last_action = raw_action.copy()

                # 动作缩放与映射
                scaled_action = raw_action.copy()
                scaled_action[[0, 3, 6, 9]] *= 0.5 
                scaled_action *= 0.25              
                target_q = Cfg.default_dof_pos + scaled_action

            # --- 底层控制 (200Hz) ---
            # 1. 获取当前真实的物理角度并重排对齐
            current_q = remap_mujoco_to_net(d.qpos[-12:])
            current_dq = remap_mujoco_to_net(d.qvel[-12:])
            
            # 2. 计算扭矩 (此时 target_q 和 current_q 都是 FR, FL, RR, RL 顺序)
            tau = Cfg.kp * (target_q - current_q) + Cfg.kd * (0 - current_dq)
            tau = np.clip(tau, -Cfg.tau_limit, Cfg.tau_limit)
            
            # 3. 直接下发！因为 xml 里的 <actuator> 也是 FR, FL, RR, RL 顺序
            d.ctrl = tau

            # --- 物理步进与渲染同步 ---
            mujoco.mj_step(m, d)

            viewer.sync() 
            
            time_until_next_step = Cfg.sim_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print("仿真结束。")