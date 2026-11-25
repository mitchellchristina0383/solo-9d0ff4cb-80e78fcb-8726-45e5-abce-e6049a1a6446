# 流式线性回归标定（Flask + SQLite，CPU）

面向小内存机器的校准训练网页：上传两列 CSV（实测值 x、参考值 y），按原行序分组，
逐段读入、累积梯度，每组只做一次带动量更新；随时可暂停，重启后可调小读取上限继续，
跑完结果与不中断训练一致。

## 快速开始

```bash
docker compose up --build -d
# 打开 http://localhost:8000
docker compose logs -f            # 查看运行日志
docker compose down               # 停止（数据在 calibration-data 卷里保留）
```

数据卷 `calibration-data` 挂到容器 `/data`：SQLite 库与上传的 CSV 都在其中，
重新 `up` 后可以直接在网页上“继续训练”。

可选环境变量（写在 `.env` 或 shell 里）：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `APP_PORT` | 8000 | 宿主机端口 |
| `APP_SECRET_KEY` | local-compose-secret | Flask 会话密钥 |
| `APP_MAX_UPLOAD_MB` | 512 | 上传大小上限 |
| `APP_MEM_LIMIT` | 512m | 容器内存硬上限 |
| `APP_CPUS` | 1.0 | 容器 CPU 配额 |

## 使用流程

1. 首页上传 CSV，填每组样本数、单次读入行数上限、学习率、动量、初始 w/b。
   表头可有可无（`x,y`、`实测值,参考值` 均可）；空行跳过。
2. 训练详情页可设“读完 N 段后暂停”或“完成 N 组更新后暂停”（留空则跑到结束），点开始。
   “段”是单次读入上限对应的块，“组”对应一次参数更新。
3. 需要主动暂停时点“主动暂停”：当前读取段结束后立刻落盘暂停，**下次从下一行继续**。
4. 暂停状态可把“单次读入行数上限”改小（例如 200 → 3）再“继续训练”。
5. 页面实时显示更新次数、已计入样本数、组内已读未更新行数、w/b、动量 v_w/v_b、
   峰值驻留行数与进程峰值 RSS，下方是可复核的逐次更新审计表。

## 算法与内存

模型 `ŷ = w·x + b`，平方误差。每组连续 `group_size` 行（末组不足也更新）：

```
g = mean(2·(w·x + b − y)·[x, 1])
v = momentum·v − lr·g
w, b += v
```

每次读取最多把 `chunk_limit` 行放进临时列表，逐行累加进组内梯度/损失，随即丢弃该列表。
峰值驻留样本数为 `min(chunk_limit, group_size)`，与文件大小无关；模型状态只有几个 double。
每行读取有 1 MiB 硬上限，畸形超长行会报错而不是吃光内存。

## 断点与一致性

SQLite 以 WAL + `synchronous=FULL` 运行；参数、动量、已计入样本数、更新次数和
组内累加器都在同一行。两种提交：

* **更新提交**（每组一次）：参数、动量、`samples_done`、`updates_done` 与一条审计记录
  在同一事务里写入 —— 一次更新要么完整生效，要么完全不生效。
* **暂停提交**（仅主动暂停，发生在读取段边界）：把组内部分梯度/损失与文件偏移
  `next_offset` 一并写入 —— 主动暂停后从下一行继续，**不重读任何行**。

意外退出不会执行暂停提交，因此从最近一次组边界重启：已计入的组绝不重读，
只有未完成的那一组重读一次（≤ 1 组）。启动时 `recover()` 把残留的 `running`
标成“意外中断”，网页上点“继续训练”即可。

浮点一致性：每行按文件顺序逐行累加进运行和，SQLite 存 IEEE-754 double，
所以读取段大小变化、暂停次数多少都不会改变舍入，续训与不中断结果逐位相同。

## 目录

```
app/            Flask 页面、上传与后台线程（单进程，训练线程 + SQLite）
trainer/        引擎（CSV 扫描、分段读取、更新、恢复、审计）与 CLI
scripts/        gen_data.py 造数；verify.py 两种中断的对照验证
tests/          unittest：分组/内存/暂停/崩溃/网页流程
```

## 验证

容器内跑全部单元/网页测试（镜像里已带 `tests/`）：

```bash
docker compose build
docker compose run --rm --entrypoint python calibrator -m unittest discover -s tests -v
```

或本机（Python 3.11+，先 `pip install -r requirements.txt`）：

```bash
python3 -m unittest discover -s tests -v
```

当前共 35 个用例，覆盖：分组与末组不足更新、空行/无表头/坏数据、
`chunk_limit` 内存上界、逐位等于不中断参考训练、N 段后暂停、暂停按钮、
调小读取上限续训、组中注入崩溃、子进程 `SIGKILL` 硬杀后续训、
网页上传/暂停/删除/重启恢复等。

两种中断的端到端对照（生成数据后各跑一次，结果逐位对比不中断训练）：

```bash
python3 scripts/gen_data.py /tmp/cal.csv -n 50000
python3 scripts/verify.py --mode pause --data /tmp/cal.csv --work /tmp/v-pause
python3 scripts/verify.py --mode kill  --data /tmp/cal.csv --work /tmp/v-kill
```

实测输出（50,000 行 / 每组 2,000 / 每段 100）：

| 模式 | 中断点 | 续训后更新/样本 | 与不中断最大差 | 审计 |
| --- | --- | --- | --- | --- |
| `none` | — | 25 / 50,000 | 0（逐位相同） | 通过 |
| `pause` | 2 组 + 200 行部分组已落盘 | 25 / 50,000 | 0（逐位相同） | 通过 |
| `kill` | 2 组已提交，组中 `os._exit(9)`；重启后段上限 100→1 | 25 / 50,000 | 0（逐位相同） | 通过 |

`pause` 校验“部分组累加器已持久化、偏移恰好对应只读一次的行数”；
`kill` 校验“只剩整组边界、未完成组回退、最多重读一组”。

内存实测（2,000,000 行 / 38 MB CSV / 每组 100,000 / 每段 500）：
子进程峰值 RSS ≈ 20.5 MB，与 50,000 行时的 ≈ 20 MB 基本相同；
数据库里记录的 `peak_rows_held = 500`。纯 Python 解析约 29 万行/秒（预热后），
200 万行整轮约 20 秒。

