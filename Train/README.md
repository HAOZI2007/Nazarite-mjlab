# Train

训练相关代码位于 [Nazarite](Nazarite/)。

- `Nazarite/mjlab/`：通用 MuJoCo 强化学习基础库；
- `Nazarite/.venv/`：当前已安装的开发环境；
- `Nazarite/Nazarite-src/nazarite/`：Nazarite 自定义训练包；
- `Nazarite/DEPENDENCIES.md`：目录依赖、任务注册链路和开发顺序。

基础库环境检查：

~~~bash
cd Nazarite/mjlab
uv sync
uv run pyright -p pyproject.toml src/mjlab
~~~

在 Nazarite 项目根目录运行训练或播放：

~~~bash
cd Nazarite
uv run list-envs
uv run train Nazarite-Velocity-Flat-Go2
uv run train Nazarite-Velocity-Flat-Go2-WTW
uv run play Nazarite-Velocity-Flat-Go2-WTW --checkpoint_file <model.pt>
~~~

当前任务均使用 Grid Adaptive 速度课程；WTW 任务额外使用行为命令和四足 phase 参考。使用说明和调参资料位于仓库根目录的 [docs](../docs/)。
