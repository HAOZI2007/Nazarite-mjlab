# 外接 rsl_rl 库教程

## 概述

[rsl_rl](https://github.com/leggedrobotics/rsl_rl) 是 ETH Zurich Robotic Systems Lab
与 NVIDIA 联合开发的强化学习库，提供 GPU 加速的 PPO 算法实现，被 Isaac Lab、
MuJoCo Playground、mjlab 等项目用作训练后端。

默认情况下 mjlab 通过 PyPI 安装 `rsl-rl-lib`，依赖锁定在固定版本。将 rsl_rl
以独立文件夹接入项目后，可以：

- 在项目中直接浏览和修改 rsl_rl 源码
- 修改即时生效，无需重新安装（editable 模式）
- 将定制改动纳入版本控制，便于复现和协作
- 方便在调试时跟踪 rsl_rl 内部逻辑

### 原理

利用 uv 的 `[tool.uv.sources]` 机制，将依赖解析从 PyPI 重定向到本地路径，
并以 **editable** 模式安装——uv 生成一个动态导入 finder（`.pth` 文件），
让 Python 直接从本地源码目录加载模块。

### rsl_rl 版本与包名对照

| GitHub tag | PyPI 包名 | 说明 |
|---|---|---|
| `v5.4.0` | `rsl-rl-lib==5.4.0` | 当前 mjlab 使用版本 |
| `v5.4.2` | `rsl-rl-lib==5.4.2` | 最新发布版本 |

---

## 前置条件

- 项目使用 **uv** 作为包管理器
- 网络可访问 `https://github.com/leggedrobotics/rsl_rl`

---

## 步骤一：克隆 rsl_rl 到项目目录

```bash
# 在项目根目录下执行
mkdir -p third_party
git clone --depth 1 --branch v5.4.0 \
  https://github.com/leggedrobotics/rsl_rl \
  third_party/rsl_rl

# 删除 .git 目录，将其作为普通文件纳入项目版本控制
rm -rf third_party/rsl_rl/.git
```

> **选择版本**：将 `v5.4.0` 替换为目标版本 tag。可通过以下命令查看可用 tag：
> ```bash
> git ls-remote --tags https://github.com/leggedrobotics/rsl_rl.git
> ```

完成后目录结构如下：

```
third_party/rsl_rl/
├── pyproject.toml        # rsl_rl 自身构建配置（含静态版本号）
├── setup.py
├── README.md
├── LICENSE
├── rsl_rl/               # 实际 Python 包
│   ├── __init__.py
│   ├── algorithms/       # PPO, Distillation
│   ├── env/              # VecEnv 接口
│   ├── extensions/       # RND, Symmetry
│   ├── models/           # MLPModel, CNNModel, RNNModel
│   ├── modules/          # CNN, MLP, Distribution, Normalization
│   ├── runners/          # OnPolicyRunner, DistillationRunner
│   ├── storage/          # RolloutStorage
│   └── utils/            # Logger, 工具函数
└── tests/
```

> **注意**：rsl_rl 是纯 Python 包，无编译扩展，可直接使用。

---

## 步骤二：修改 `pyproject.toml`

### 2.1 修改依赖声明

将固定版本号改为无版本约束（版本由 vendored 源码决定）：

```toml
# 修改前
dependencies = [
  ...
  "rsl-rl-lib==5.4.0",
  ...
]

# 修改后
dependencies = [
  ...
  "rsl-rl-lib",
  ...
]
```

### 2.2 添加 source override

在 `[tool.uv.sources]` 表中添加本地路径：

```toml
[tool.uv.sources]
rsl-rl-lib = { path = "third_party/rsl_rl", editable = true }
```

> **editable = true**：修改 `third_party/rsl_rl/` 下的源码后即时生效，无需重新 `uv sync`。

完整示例（与现有 source 合并）：

```toml
[tool.uv.sources]
warp-lang = { index = "nvidia", marker = "sys_platform != 'darwin'" }
rsl-rl-lib = { path = "third_party/rsl_rl", editable = true }
torch = [
  { index = "pytorch-cu128", extra = "cu128", marker = "sys_platform != 'darwin'" },
  { index = "pytorch-cpu", extra = "cpu", marker = "sys_platform != 'darwin'" },
]
```

---

## 步骤三：配置类型检查工具

Editable 安装通过动态 finder 机制（`.pth` 文件）加载模块，部分类型检查器
可能无法自动识别，需要额外配置搜索路径。

### 3.1 ty (astral-sh/ty)

```toml
[tool.ty.environment]
extra-paths = ["typings", "third_party/rsl_rl"]
```

### 3.2 pyright

```toml
[tool.pyright]
ignore = [..., "./third_party"]
extraPaths = ["third_party/rsl_rl"]
```

> **说明**：`ignore` 排除 `third_party/` 避免 pyright 检查 vendored 代码的内部类型问题；
> `extraPaths` 确保导入解析能找到 `rsl_rl` 包。

### 3.3 .gitignore

添加 build 产物目录（uv sync 时可能生成）：

```gitignore
third_party/rsl_rl/build/
```

---

## 步骤四：锁定和同步依赖

```bash
uv lock   # 重新生成 uv.lock，将 rsl-rl-lib 的 source 从 PyPI registry 改为本地 path
uv sync   # 安装 vendored 版本
```

> **预期输出**：`uv sync` 会显示卸载旧的 PyPI 版本、安装新的本地版本：
> ```
>  - rsl-rl-lib==5.4.0
>  + rsl-rl-lib==5.4.0 (from file:///<project>/third_party/rsl_rl)
> ```

**rsl_rl 的传递依赖**（torchvision, GitPython, onnx 等）会由 uv 自动解析，
无需手动添加到项目直接依赖中。

---

## 步骤五：验证

### 5.1 确认导入路径

```bash
uv run python -c "import rsl_rl; print(rsl_rl.__file__)"
```

期望输出指向 vendored 目录：
```
/path/to/your_project/third_party/rsl_rl/rsl_rl/__init__.py
```

### 5.2 确认核心模块可导入

```bash
uv run python -c "
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.models import MLPModel, CNNModel
from rsl_rl.modules import CNN, MLP
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
print('All imports OK')
"
```

### 5.3 验证 editable 模式

在 `third_party/rsl_rl/rsl_rl/__init__.py` 末尾添加临时测试行：

```python
# __TEST__ = "editable works"
```

然后在 Python 中验证：

```bash
uv run python -c "import rsl_rl; print(rsl_rl.__TEST__)"
# 输出: editable works
```

确认后删除临时行。

### 5.4 运行测试和类型检查

```bash
uv run pytest tests/       # 运行测试套件
uv run ty check            # ty 类型检查
uv run pyright             # pyright 类型检查
```

---

## 日常使用与维护

### 修改 rsl_rl 源码

直接编辑 `third_party/rsl_rl/rsl_rl/` 下的文件，修改会即时生效。

### 升级到新版 rsl_rl

```bash
cd third_party/rsl_rl
# 方法一：重新 clone 新版本
cd .. && rm -rf rsl_rl
git clone --depth 1 --branch v5.4.2 \
  https://github.com/leggedrobotics/rsl_rl rsl_rl
rm -rf rsl_rl/.git rsl_rl/build

# 方法二：如果要保留本地修改
git init && git remote add origin https://github.com/leggedrobotics/rsl_rl
git fetch --tags origin
git diff v5.4.0..v5.4.2  # 对比差异

cd /path/to/project
uv lock && uv sync
```

### 提交到版本控制

`third_party/rsl_rl/` 作为项目的一部分，直接纳入 git 版本控制即可。
`.gitignore` 中已排除 `build/` 产物目录。

### 检查本地改动

```bash
git diff -- third_party/rsl_rl/
```

---

## 常见问题

### Q: `uv sync` 报版本检测错误

如果 rsl_rl 的 `pyproject.toml` 使用 setuptools-scm 动态检测版本（依赖 git tag），
且 `.git` 目录已被删除，设置静态版本号即可：

```toml
# third_party/rsl_rl/pyproject.toml
[project]
name = "rsl-rl-lib"
version = "5.4.0"  # 确保此行存在
```

当前 (v5.4.0) 已包含静态版本号，无需额外处理。

### Q: `ty check` 报 `unresolved-import` 找不到 `rsl_rl`

确保 `[tool.ty.environment]` 中 `extra-paths` 包含 `"third_party/rsl_rl"`。

### Q: `pyright` 报大量 vendored 代码内的类型错误

确保 `[tool.pyright]` 的 `ignore` 列表中包含 `"./third_party"`。

### Q: 想回退到 PyPI 版本

```bash
# 1. 恢复 pyproject.toml 中的依赖声明
#    "rsl-rl-lib" → "rsl-rl-lib==5.4.0"
# 2. 删除 [tool.uv.sources] 中的 rsl-rl-lib 行
# 3. 删除 vendored 目录
rm -rf third_party/rsl_rl
# 4. 重新锁定和同步
uv lock && uv sync
```

### Q: 能否用 git submodule 代替 vendoring

可以，但需要注意：
- submodule 需要单独同步和管理
- 修改 submodule 代码需要向独立仓库提交
- uv 的 `path` 同样支持 submodule 目录

如果不需要修改 rsl_rl 源码，且希望更清晰的更新流程，submodule 是更好的选择。
如果需要频繁修改和定制，纯 vendoring 更灵活。

---

## 参考

- [rsl_rl GitHub 仓库](https://github.com/leggedrobotics/rsl_rl)
- [uv 依赖源文档](https://docs.astral.sh/uv/concepts/dependencies/#dependency-sources)
- [rsl_rl 论文](https://arxiv.org/abs/2509.10771)
