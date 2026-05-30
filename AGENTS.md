## Language requirements
- 思考过程、debug过程可以用英文，但是最终结论里优先用中文回答。对于所有专业名词、技术词语，请在括号内标注英文。
- Update project level AGENTS.md (in project root directory) on demand at any time.

## Unitree G1 MuJoCo environment
- 本项目的 Unitree G1 / MuJoCo 开发环境默认用 uv（uv）管理；优先使用 `uv run ...`，不要随手切到系统 Python。`pyproject.toml` 要求 Python `>=3.12,<3.13`，当前 `.python-version` 是 `3.12`。
- 已知坑：Homebrew Python 3.13/3.14 与 `cyclonedds==0.10.2`/Unitree SDK2 Python 兼容性不好；之前 Python 3.13 编译 CycloneDDS Python 绑定失败，Homebrew Python 3.12 也出现过 `pyexpat` 动态库链接问题。需要重建环境时，优先让 uv 提供 CPython 3.12。
- 关键依赖版本：MuJoCo（MuJoCo）`3.9.0`、pygame（pygame）`2.6.1`、CycloneDDS Python（CycloneDDS Python）`0.10.2`、Unitree SDK2 Python（unitree_sdk2py）来自本地 `unitree_sdk2_python` editable source。
- CycloneDDS C 库（CycloneDDS C library）安装在项目内 `cyclonedds/install`；运行脚本必须设置 `CYCLONEDDS_HOME=/Users/eric/Projects/PhysicalAI/cyclonedds/install`。
- macOS Apple Silicon 上 DDS 网卡（DDS interface）使用 `lo0`，不是 Linux 文档里的 `lo`。仿真默认 `DOMAIN_ID = 1`，避免和真机常用 domain `0` 混在一起。
- G1 基础仿真配置来源是 `configs/unitree_mujoco_g1_config.py`，运行 `scripts/run_unitree_g1_mujoco.sh` 时会同步到 `unitree_mujoco/simulate_python/config.py`。关键值应保持：`ROBOT = "g1"`、`ROBOT_SCENE = "../unitree_robots/g1/scene_29dof.xml"`、`INTERFACE = "lo0"`、`USE_JOYSTICK = 0`。
- 基础环境检查命令：`uv run python scripts/check_unitree_g1_setup.py`。G1 29DOF 模型的正常加载参考输出是 `actuators: 29, nq: 36, nv: 35`。
- Unitree 官方 `unitree_mujoco/simulate_python/test/test_unitree_sdk2.py` 默认发送 Go2 消息（unitree_go message），不要直接拿它测试或控制 G1。G1 低层控制应使用 `unitree_hg` 消息（unitree_hg message），关节顺序参考 `unitree_mujoco/unitree_robots/g1/g1_joint_index_dds.md`。
- `unitree_mujoco`、`unitree_sdk2_python`、`cyclonedds` 是本地固定依赖/上游 checkout（upstream checkout），通常不纳入 Git；根仓库主要记录新开发的 demo、脚本、配置和文档。macOS 大小写不敏感文件系统上 `unitree_mujoco` 可能提示 `terrain.STL`/`terrain.stl` 冲突，主要影响 go2w 地形资源，不应误判为 G1 blocker。

## G1 safety and control assumptions
- 当前项目所有 G1 动作 demo 默认先做 MuJoCo 仿真（simulation）和离线运动学播放（kinematic playback），不能直接把生成轨迹当作真机低层控制（real robot low-level control）执行。
- 真机前必须补齐：轨迹平滑（trajectory smoothing）、速度/加速度限制（velocity/acceleration limits）、力矩限制（torque limits）、足底/质心稳定检查（foot/COM stability checks）、碰撞检查（collision checking）和急停策略（emergency stop strategy）。
- 当前 `g1_29dof.xml` 的 rubber hand 是刚性网格（rigid mesh），没有手指关节（finger joints）。抓握只能通过整只手腕姿态（wrist orientation）和被动 adapter（passive adapter/fixture）模拟，不能假装手指能主动弯曲握紧。
- G1 基础模型本体有 29 个可控关节自由度（actuated joint DOF）。当前 mochitsuki 场景额外包含杵/锤子的 free joint（free joint）；如果讨论完整 MuJoCo scene 的 `nq/nv`，要先确认是否加载了 `kine_hammer` 等额外 body。

## Mochitsuki G1 hard rule
- 在 `demos/mochitsuki` 的 Unitree G1 餅つき demo 中，左右手抓握不能反：`left_wrist_yaw_link` 必须在 G1 解剖左侧（anatomical left）握杵，`right_wrist_yaw_link` 必须在 G1 解剖右侧（anatomical right）握杵。
- 当前 hammer frame（杵坐标系）里，杵局部 `+X` 约等于 G1 解剖左侧，局部 `-X` 约等于 G1 解剖右侧。因此左手 adapter 必须是 `adapter_side_local = [1, 0, 0]`，右手 adapter 必须是 `adapter_side_local = [-1, 0, 0]`；不能为了数值好看把这两个符号互换。
- 如果渲染图里出现左手握右侧、右手握左侧，必须视为 blocker，先修正左右手侧别，再继续优化姿态、穿模、速度或稳定性。
- 如果改成参考视频里的 roll drill（锤柄横在胸腹/髋前低位，锤头在身体一侧上下提落），旧的 hammer local `+X/-X` 规则不再等价于左右手侧别。此时必须用两个抓握中心在 G1 解剖左轴（anatomical-left axis）上的绝对投影判断：左手抓握中心必须始终在右手抓握中心的解剖左侧；adapter 可以按视频动作从身体侧包住锤柄。
- 对当前多角度参考视频，握点必须是一手接近杆尾、一手在杆子中段附近，不能把两手挤在中段；rubber hand 的可视姿态必须优先表现“掌心面对杵柄”，adapter 用来包住掌面和杵柄，不能让杵穿过手掌或手腕侧面。
- demo 视频、关键帧和渲染验证必须默认使用多角度（multi-camera）输出，至少覆盖主视角、抓握近景、手部前/后/侧视和俯视；这些相机必须来自明显不同方位，不能只是同角度不同远近，且整套布局最多保留一个抓握特写。单相机视频只能作为临时调试，不能作为最终判断依据，尤其不能使用看不到杵柄和抓握关系的背面远景。

## Mochitsuki demo workflow
- `demos/mochitsuki/mochitsuki_demo.py` 和 `demos/mochitsuki/scene_mochitsuki.xml` 是主线迭代文件；v3 冻结版（frozen snapshot）是 `demos/mochitsuki/mochitsuki_demo_v3.py`、`demos/mochitsuki/scene_mochitsuki_v3.xml`、`scripts/render_mochitsuki_demo_v3.sh`，不要在未明确要求时改写冻结版。
- v2 已保存为 Git tag `v2` / commit `a6992b0`；如果用户要求“回到 v2”，只恢复明确相关的 mochitsuki 文件，不要碰无关脏文件。
- 杵/锤子模型应保持用户给定比例：头长约 `0.40m`、头直径约 `0.055m`、持手段约 `0.54m`、总质量约 `1.0kg`。普通餅つき节奏不是劈柴式猛砍；默认演示应偏保守，约 `1.3-1.6s` 一个周期，除非用户明确要求高速版本。
- 当前 roll drill 目标动作：锤柄低位横在胸腹/髋前，锤头在身体一侧上下提落；外侧/右手应举得相对更高，中段手高于尾端手，左手/尾端手保持腰腹高度；双上臂要贴近身体，更多靠肘关节挥动，不要大幅外展甩肩。
- 臼和饼的位置应在机器人略偏右侧，使宽站姿更自然；头/躯干应通过腰偏航/俯仰（waist yaw/pitch）看向落点。G1 没有独立头颈关节时，用上身朝向表达“看目标”。
- 每次修改 mochitsuki 动作后按顺序验证：`uv run python demos/mochitsuki/mochitsuki_demo.py --mode check --render-smoke`，再 `--mode render --camera-layout multi --save-keyframes`，最后 `--mode diagnose --diagnostic-samples 52`。必须实际查看多角度渲染图，不能只看数值指标。
- 动态 warning（joint velocity/acceleration、estimated torque ratio、elbow hyperextension）不能忽略；如果为了保持抓握或姿态导致 warning 升高，应在结论里明确说明这是仿真/视觉原型，不是可上真机轨迹。
