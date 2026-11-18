# 机器人互动摄影 · Robot Portrait Agent

**作者：邓羽希（chloe）**

将商家吸引路人驻足的需求，做成可观看、可参与、可带走照片的机器人摄影体验。系统以 **Claude 多模态构图 + MediaPipe 本地跟踪 + UR 机械臂反馈控制**，串联邀请、引导、倒计时、拍摄和临时分享。

这是一个可以从接口、交互状态、控制逻辑和测试用例逐层查看的 AI 产品原型。仓库同时保留交付版与多轮 Agent 实验实现，明确标注两者的能力边界。

## 1. 产品问题与方案

**问题：** 普通展示吸引注意后，缺少让路人主动参与和带走内容的环节。

**方案：** 在门店橱窗或品牌活动入口设置机器人摄影体验。动作和灯光让过程可被围观；顾客点击开始，由系统给出构图与位置引导，到位后拍摄并提供下载二维码。

```text
注意并进入 → 主动开始 → 构图规划 → 方向引导 → 到位与倒计时 → 拍照与下载
                              ↑          ↓
                       返回参考位 ← 限幅动作
                              ↓
                         新画面重新观察
```

产品取舍：一次构图规划后锁定目标，由本地视觉持续判断是否到位，减少重复模型调用与目标漂移。允许停止；公开分享需单独同意。驻足率、体验开始率、拍摄完成率和下载率是后续门店验证指标，当前没有商业转化增长数据。

## 2. 技术架构

| 层 | 实现 | 负责什么 |
| --- | --- | --- |
| 采集与界面 | iPad 前置相机、HTML/JavaScript、配对 HTTPS | 画面上传、开始/停止、灯光提示、倒计时和结果页 |
| 构图规划 | Anthropic Messages API、JSON Schema | 从图像及人物几何信息返回构图目标与理由 |
| 本地视觉 | MediaPipe 0.10.21、OpenCV 4.11.0.86、NumPy 1.26.4 | 头部位置、表观肩宽、可见性及构图可行性检查 |
| 控制与反馈 | Python 3.11–3.12、URScript、ur-rtde 1.6.5 | UR5e/UR10e 限幅动作、静止验证、参考位和会话生命周期 |
| 提示与交付 | 预生成 WAV、临时 JPEG、随机下载链接 | 多模态引导及最多 300 秒的照片分享 |

模型输出是图像坐标中的目标，不是机器人路径或关节指令。动作仍需经过本地状态、几何和控制反馈检查。

## 3. AI 如何接入

1. `scripts/configure-claude.py` 通过隐藏回显读取密钥，写入本地 `.env.local`；客户端读取 `ANTHROPIC_API_KEY`，不把密钥提交到仓库。
2. `ClaudeFramingClient.check_access()` 请求模型信息，先验证访问权限，不发送相机图像。
3. `analyze()` 将 JPEG 编码为 Base64，连同 `current_pose` 和可选 `measured_context` 提交到 `https://api.anthropic.com/v1/messages`。
4. 系统提示词限定镜像全图坐标、可见人物和构图边界；图像中的文字不是控制指令。`output_config` 使用 JSON Schema 约束返回值。
5. 本地解析并验证字段、有限数值、置信度与几何关系，质量模式再按人物边距修正或拒绝不合理构图。低置信度、无效响应和不可行目标不能触发运动。
6. 目标锁定后由本地视觉持续跟踪；机械臂返回参考位、稳定后读取新画面，避免把相机移动误认为人移动。

核心结果字段：

| 字段 | 含义 |
| --- | --- |
| `subject_visible` / `confidence` | 人物可见性与几何可靠程度 |
| `target` | 归一化的头部横坐标、纵坐标、表观肩宽 |
| `frame` | 全图中的裁切边界 `[left, top, right, bottom]` |
| `style` | 头肩、半身或环境人像构图 |
| `explanation` | 简洁构图理由，仅用于展示 |

归档代码默认配置为 `claude-opus-5-5`、`medium`；这是包内配置记录，不代表任意账户当前均可调用。运行前需验证实际模型权限，并使用 `--claude-model` 配置可用模型。

## 4. 两种交互模式

| | 交付流程 `claude-guide + single-plan` | 实验分支 `claude-agent` |
| --- | --- | --- |
| 决策方式 | 一次参考画面构图，之后本地闭环 | 按事件观察并选择下一步交互 |
| 上下文 | 当前图像、人物位置及几何信息 | 最多 4 帧、最多 16 条历史、上一动作结果、人物状态 |
| 行为 | 预设方向提示、等待、到位与拍照 | 邀请、提议构图、引导、等待、确认或结束 |
| 请求控制 | 明确开始或重新构图才建立新目标 | 至少间隔 5 秒，默认总请求上限 20 次 |
| 过期处理 | 会话与目标状态检查 | 会话代次、时效和人物变化共同过滤旧回复 |

实验 Agent 的代码与测试可供查看，但不将它描述为已完成所有现场验收的交付能力。交付版不启用对话式麦克风输入，方向提示由预生成音频播放。

## 5. 调试与评测方法

- **相机运动造成误判：** 增加返回参考位、静止门控和新画面检查；分别验证方向动作和观察时点。
- **停止后旧回复重放：** 使用会话代次使过期异步响应失效，测试停止、重新进入和延迟回复。
- **模型结果不可信：** 用模拟响应覆盖低置信度、非法坐标、拒答、截断与服务错误；先拒绝，再反馈原因。
- **方向与实体安装不一致：** UR5e 和 UR10e 使用各自配置；保留现场修正记录与仿真指标。
- **照片生命周期：** 测试新照片替换旧照片、300 秒到期、旧链接返回 404、停止取消延迟上传。

包内历史审计：Python **2,895 项通过、2 项跳过**；Node **191 项通过、1 项跳过**。这些是归档记录，不应当作每次整理仓库后重新运行的结果。

## 6. 目录导航

```text
README.md          项目说明与运行指南
vision_mvp/        应用代码、测试、模型、配置与启动脚本
robot_programs/     UR5e 和 UR10e 动作程序与配置
tmp/               启动器所需的固定版本运行包和版本索引
```

推荐阅读入口（相对 `vision_mvp/src/leica_tracker/`）：

- `claude_framing.py`、`portrait_quality.py`：模型请求、提示词与几何验证。
- `claude_agent.py`、`agent_actions.py`：多轮上下文、结构化动作和事件调度。
- `app.py`、`physical_runtime.py`：整体流程与机器人生命周期。
- `reference_gate.py`、`network_reference.py`：参考位、重启和网络间断处理。
- `guest.html`、`guest_display.py`：访客界面与状态同步。
- `photo_share.py`：照片上传、有效期、清除与下载。

## 7. 环境与运行

安装 Python 3.11 或 3.12、`uv`；浏览器界面测试需要 Node.js 20 以上。原交付启动脚本以 Windows / PowerShell 为目标环境。

```powershell
uv sync --project ./vision_mvp --frozen --extra robot --extra handoff --python 3.12
```

先进行离线完整性检查（不连接机器人，也不发起模型请求）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./vision_mvp/scripts/start-canonical-ipad.ps1 -RobotModel ur5e
```

另一机型使用 `ur10e`。独立运行包校验：

```bash
python vision_mvp/scripts/verify-canonical-runtime.py \
  --bundle tmp/canonical-delivery-restart34-english \
  --runtime-root .
```

配置本地 Claude 密钥并验证访问（需先激活项目环境）：

```bash
python vision_mvp/scripts/configure-claude.py
python -m leica_tracker.claude_framing
```

模型权限检查不上传相机图像。实体运行需要现场校准后的 profile、已配对 iPad、正确网络和操作人员确认；启动器提供 `-Execute`、`-Profile` 与 `-GuestSession` 参数。仓库不包含可直接套用到任意场地的安全认证或标定。

自动化测试（在项目根目录，激活项目环境后）：

```bash
python -m pytest -q -p no:cacheprovider vision_mvp/tests
node --test vision_mvp/tests/*.test.cjs
```

## 8. 数据与实现边界

- 选取的相机图像和人物几何数据发送到配置的 Claude 服务；不是全程本地推理。
- 照片仅在计算机内存保存当前一张，最大 4 MiB、最长 300 秒；下载后的副本不会被系统删除。
- 随机照片链接属于持有者链接，不是身份认证；开始体验不等于授权公开传播。
- 机械臂软件限幅不能替代固定装置、负载、急停与空间检查。仓库中的运动导出文件主要用于审阅和复现研究。

## 9. 实现与素材来源

我设计了项目需求和交互流程，并根据现场反馈调整方案。开发中使用 Codex 辅助实现、调试和测试，运行时由 Claude 提供构图建议。概念图和视频用于展示活动场景。

项目保留 Three.js 许可、MediaPipe 模型来源和 Piper 语音素材记录。
