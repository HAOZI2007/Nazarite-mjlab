# Train

训练相关代码位于 [Nazarite](Nazarite/)。

- `Nazarite/mjlab/`：通用 MuJoCo 强化学习基础库；
- `Nazarite/mjlab/.venv/`：当前已安装的开发环境；
- `Nazarite/Nazarite-src/nazarite/`：Nazarite 自定义训练包；
- `Nazarite/DEPENDENCIES.md`：目录依赖、任务注册链路和开发顺序。

基础库环境检查：

~~~bash
cd Nazarite/mjlab
uv sync
uv run pyright -p pyproject.toml src/mjlab
~~~

完成自定义任务配置后，在 Nazarite 项目根目录运行训练：

~~~bash
cd Nazarite
uv run list-envs
uv run train <task-id>
uv run play <task-id>
~~~
