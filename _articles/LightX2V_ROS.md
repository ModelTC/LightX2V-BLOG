---
layout: post
title: "LightX2V ROS：让带 Action 输出的世界模型真正闭环运行"
author: "LightX2V Team"
date: 2026-07-22
tags: [LightX2V, ROS 2, Robotics, World Models]
---

# LightX2V ROS：让带 Action 输出的世界模型真正闭环运行

一个能够输出 Action 的世界模型，并不会因为完成了一次推理，就自动变成一个可以操作机器人的系统。

真正的机器人控制是一个持续循环：模型读取当前环境的图像和机器人状态，预测动作；环境执行动作并发生变化；模型再读取新的观测，继续预测下一步动作。只要这个循环中断，模型输出的就只是一段离线轨迹，而不是一个能够与环境交互的策略。

LightX2V ROS 解决的正是这个问题。它把 LightX2V 中的视觉—动作模型、LIBERO/RoboTwin/RoboLab 仿真环境以及可视化控制面封装为相互独立的 ROS 2 节点，通过统一 Topic 构成一个可以持续运行、切换任务、统计结果的闭环系统。

![LightX2V ROS Architecture]({{ site.baseurl }}/assets/LightX2V_ROS/lightx2v_ros.png)

## 为什么带 Action 输出的世界模型必须闭环

传统视频生成模型的输入和输出通常都是一次性的。输入一张图或一段文本，模型生成完整视频，任务到此结束。

机器人策略不是这样。假设时刻 `t` 的环境观测为 `o_t`，策略根据观测预测动作：

```text
a_t = Policy(o_t, state_t, instruction)
```

动作被机器人或仿真环境执行后，环境进入新状态，并产生下一帧观测：

```text
(o_{t+1}, state_{t+1}) = Environment.step(a_t)
```

模型必须继续基于 `o_{t+1}` 进行推理。这个过程不断重复，直到任务成功或达到最大执行步数：

```text
Observe -> Infer -> Act -> Observe -> Infer -> Act -> ...
```

即使模型一次生成的是一个 Action Chunk，也不能无限期地盲目执行。执行若干动作后重新观察环境，再基于真实反馈规划新的 Action Chunk，通常比完全开环执行更可靠。物体可能滑动，机械臂可能存在执行误差，仿真中的接触状态也可能与预测视频不同；新的真实观测是修正这些误差的唯一依据。

这里所说的 Environment 可以是真实机器人、机械臂，也可以是 LIBERO、RoboTwin、RoboLab 中的仿真机器人。无论载体是什么，模型都需要同一类闭环：消费观测，输出动作，等待环境执行，再消费下一次观测。

## LightX2V ROS 的核心思路：模型和环境彼此隔离

最直接的实现方式，是在某个模型的 Python 脚本里直接创建仿真环境，然后在一个大循环中同时调用模型和环境。这种方式很快就会遇到问题：模型代码依赖 PyTorch、CUDA 和权重加载，仿真环境又分别依赖 MuJoCo、SAPIEN、IsaacSim/IsaacLab；任何一侧变化，另一侧都可能被迫修改。

LightX2V ROS 没有把所有组件塞进一个进程，而是把它们拆成独立 ROS 2 节点：

- `fastwam_node`、`lingbot_va_node`、`cosmos3_node` 负责模型推理。
- `libero_node`、`robotwin_node`、`robolab_node` 负责仿真环境执行。
- `image_web_viewer` 负责实时画面、任务状态和评测控制。
- `common.EnvContract` 负责定义双方共同遵守的接口。

推理节点不需要 import 某个仿真器，仿真节点也不需要加载模型权重。它们只通过 ROS Topic 交换标准消息。这让模型和环境可以使用不同 Python 环境、不同 GPU，甚至运行在不同机器上，只要位于同一个可通信的 ROS Domain 中即可。

### 数据面和控制面

每个环境拥有独立 namespace，例如 `/libero`、`/robotwin`、`/robolab`。核心 Topic 包括：

| Topic | 消息类型 | 作用 |
|---|---|---|
| `/<env>/<camera>/image_raw` | `sensor_msgs/Image` | RGB 相机图像 |
| `/<env>/state` | `Float32MultiArray` | 机器人 proprioception 状态 |
| `/<env>/task_description` | `String` | 自然语言任务指令 |
| `/<env>/observation_ready` | `Int32` | 新观测提交序号 |
| `/<env>/action` | `Float32MultiArray` | 模型输出动作 |
| `/<env>/episode` | `Int32` | Episode 序号，用于重置策略状态 |
| `/<env>/success` | `Bool` | 当前任务是否成功 |
| `/<env>/control` | JSON `String` | start/pause/resume/restart/set_task |
| `/<env>/status` | JSON `String` | 状态机、任务配置、历史和成功率 |

其中 `observation_ready` 是闭环同步的关键。仿真节点先发布图像、机器人状态和任务文本，最后发布单调递增的 observation index。推理节点只有看到一个尚未处理的新 index，才会读取缓存中的最新观测并产生一条动作。

动作到达仿真节点后，仿真节点执行一次 `env.step(action)`，得到新的观测并再次发布。由此形成严格的 Observation—Action 循环。周期性重发机制还能让晚启动的推理节点和 visualization 节点获得当前状态，而不会因为错过第一帧一直等待。

## 一套统一的仿真环境接口

模型与环境分进程只是第一步。更重要的问题是：三个仿真环境的原生 API、相机命名、机器人状态和动作语义完全不同。如果推理节点直接理解这些差异，那么每新增一个模型，都要为 LIBERO、RoboTwin、RoboLab 分别写一份环境代码。

LightX2V ROS 通过三层结构避免了这种重复。

### EnvContract：协议的单一来源

[`common/contract.py`](../src/common/common/contract.py) 中的 `EnvContract` 为每个环境定义：

- namespace 和所有派生 Topic 名称；
- 环境能够发布的全部相机；
- 真正输入策略的相机子集及固定顺序；
- Action 和 State 维度；
- 建议图像尺寸；
- Policy Profile、归一化模式和夹爪后处理方式。

推理、仿真和 visualization 节点都读取同一份 Contract，因此不会分别硬编码 `/libero/action` 或相机列表。例如 viewer 默认展示 `contract.cameras` 中的全部相机，而推理节点只订阅 `contract.policy_input_cameras`。

当前契约如下：

| 环境 | Policy 输入相机 | State | Action |
|---|---|---:|---:|
| LIBERO | `agentview`, `wrist` | 8 | 7 |
| RoboTwin | `head_camera`, `left_camera`, `right_camera` | 14 | 14；兼容 LingBot-VA 的 16 维 EE Action |
| RoboLab | `wrist_cam`, `over_shoulder_left_camera`, `over_shoulder_right_camera` | 8 | 8 |

### BaseSimEnv：屏蔽仿真器原生 API

[`sim/base_env.py`](../src/simulator/simulator/sim/base_env.py) 定义了环境无关接口：

```python
reset() -> Observation
step(action) -> tuple[Observation, bool]
new_episode() -> Observation
```

统一的 `Observation` 只有两个字段：

```python
images: dict[str, np.ndarray]  # H x W x 3 RGB uint8
state: np.ndarray              # 一维 float32 向量
```

每个具体环境只需要把自己的原始输出转换成这个结构，并把统一动作转换回原生控制接口。

LIBERO 适配器将末端位置、四元数转换后的 axis-angle 和夹爪 qpos 拼成 8 维状态，同时修正相机图像方向；7 维动作直接交给 MuJoCo 环境执行。

RoboTwin 适配器输出 14 维关节状态，并兼容两类动作：FastWAM 的 14 维绝对 qpos，以及 LingBot-VA 的 16 维双臂相对末端位姿。后者会在环境边界转换成绝对 EE target，再交给 SAPIEN 执行。模型节点不需要包含任何 SAPIEN 控制代码。

RoboLab 适配器负责 IsaacSim/IsaacLab 的启动顺序、Tensor 与 NumPy 转换、相机 batch 维处理，并把 `arm_joint_pos + gripper_pos` 统一成 8 维状态。Cosmos3 输出的 8 维 Action 会被转换成设备上的 batched Tensor 后执行。

### SimulatorNode：一份 ROS 循环，三个运行实例

[`sim/node.py`](../src/simulator/simulator/sim/node.py) 中的 `SimulatorNode` 实现统一的 ROS 发布订阅、Episode 管理和评测状态机。它通过构造参数接收 `contract` 和 `env_factory`：

```python
SimulatorNode(contract, env_factory, node_name=...)
```

这是一种组合和依赖注入设计。`SimulatorNode` 内部持有 `self.env`，但不知道它究竟是 `LiberoEnv`、`RoboTwinEnv` 还是 `RoboLabEnv`。三个启动入口创建的是三个配置不同的 `SimulatorNode` 实例，而不是再额外运行一个通用 simulator 节点。

状态机支持 `ready`、`running`、`paused`、`success`、`failure` 和 `switching`。它还统一处理最大步数、连续评测、任务切换、Episode 历史以及中间 viewer 帧发布。

这种设计带来的直接收益是：新增模型时，模型只需要适配 Contract 定义的 ROS 输入输出，不必为每个仿真环境重写执行循环；新增仿真环境时，也只需要实现新的 `BaseSimEnv` 和 Contract，通用 `SimulatorNode` 无需修改。

## ROS 节点如何调用真正的 LightX2V 推理

三个推理节点是对外 ROS 接口，真正的模型加载、编码器、VAE、Scheduler 和 Action Chunk 生成仍然由 LightX2V Runtime 完成。

ROS 节点没有通过 shell 反复启动 `python -m lightx2v.infer`，而是直接 import LightX2V runner 模块中的长期驻留 Policy 包装器。模型权重只加载一次，后续每个 observation index 调用 `next_action()`。

### FastWAM

`fastwam_node` 使用 `FastWAMPolicy`。策略根据 Contract 选择相机，LIBERO 使用 `agentview | wrist` 拼接，RoboTwin 使用 head 和双 wrist 的 T 形布局。图像经过 Wan VAE，任务文本经过 T5，机器人状态经过数据集统计量归一化，然后由 `FastWAMNativeModel` 生成 Action Chunk。

策略只把配置中的前 `actions_per_plan` 个动作加入队列。ROS 每收到一次新观测弹出一个动作；队列耗尽后，再用最新真实观测重新规划。

### LingBot-VA

`lingbot_va_node` 创建真正的 `LingbotVARunner`，但将其切换为在线闭环模式。首次观测用于初始化文本编码、Streaming VAE 和 KV Cache；模型联合生成视频和 Action Chunk。

执行 Action Chunk 的过程中，节点按 VAE 时间步长收集真实相机帧。Chunk 执行完后，`update_online_cache()` 会用真实观测和真实已执行动作替换预测历史，再生成下一个 Chunk。这是 LingBot-VA 从离线视频—动作生成切换到闭环策略的核心。

### Cosmos3

`cosmos3_node` 使用 `Cosmos3Policy` 和真实的 `Cosmos3Runner`。RoboLab 的 wrist 和两个 shoulder view 会被拼成 Policy-DROID 需要的布局，8 维机器人状态作为 Action History 条件输入模型。

对于多卡配置，只有 rank 0 参与 ROS DDS 通信。rank 0 将完整观测以及 step/reset/stop 命令广播给其他 torchrun rank，所有 rank 同步进入模型 collective，只有 rank 0 把最终动作发布给 RoboLab。这样既保留 LightX2V 的并行推理能力，又不会在 ROS Graph 中出现重复节点和重复动作。

## 可视化节点：不仅看画面，也控制评测

[`image_web_viewer_node`](../src/visualization/visualization/image_web_viewer_node/main.py) 会根据 EnvContract 自动订阅当前环境的全部相机，把 ROS Image 编码成 JPEG，通过多线程 HTTP Server 输出 MJPEG Stream。

浏览器页面不仅展示画面，还会读取 `/status` 中的状态机、任务、seed、Episode 进度和历史成功率。Start、Pause、Resume、Restart、Set Task 等操作通过 HTTP `POST /control` 转换成 ROS control JSON，再由通用 `SimulatorNode` 执行。

RoboTwin 在执行单条 Action 时可能包含很多物理仿真步。它可以通过 frame callback 发布中间画面，但不会发布新的 `observation_ready`，因此 viewer 能看到连续运动，而策略不会错误地重复推理。
