# 基础验证示例
python setup.py build_ext --inplace  # 编译安装
export ASCEND_RT_VISIBLE_DEVICES=5  # 指定pytest运行使用的NPU卡号，优先选择空闲NPU
pytest -rs -v tests/test_flash_attn_npu_v4.py  # 仓库用例验证
pytest -rs -v tests/test_flash_attn_npu_v4_graph.py  # 仓库用例验证
pytest -rs -v tests/test_flash_attn_npu_v4_metadata.py  # 仓库用例验证
pytest -rs -v tests/test_torch_compile_fa4.py  # 仓库用例验证

<!-- pytest -rs -v tests/test_flash_attn_npu_v4_graph.py tests/test_flash_attn_npu_v4_metadata.py tests/test_torch_compile_fa4.py tests/test_flash_attn_npu_v4.py  -->

# atk用例验证（在线双标杆精度，裸ATK命令）

精度工程在 [`tests/atk/atk-precision`](tests/atk/atk-precision/README.md)，可以完整拷贝到仓库外，例如两端均部署到 `/srv/fa4-precision`，无需设置仓库路径或 `PYTHONPATH`。性能工程 `tests/atk/atk-perf` 不变。

同一个三合一执行器按 `self.device` 分派：NPU FA4 待测接口、GPU 官方 CUDA FA4 同精度标杆、CPU FP64 数学真值。两端扩展按需导入，不交叉依赖。现成 `cases/*.json` 各100例，显式使用 `double_benchmark`，无需先跑准备脚本。

## GPU端启动服务

选择空闲GPU，在GPU服务器执行：

```bash
CUDA_VISIBLE_DEVICES=0 atk server \
  --devices 0 --port 9090 --timeout 3600 \
  --plugin_path /srv/fa4-precision/executor_flash_attn_fa4.py
```

需安装官方 `flash-attn-4`/CuTe 和 CUDA PyTorch；普通FA2包不能替代。**server 的 `-p` 是端口，插件应写完整 `--plugin_path`。**GPU端先加载同一执行插件，服务仅开放在可信测试网络；两端ATK版本一致（本工程按26.9.24核对）。容器需要映射服务端口，按环境配置代理绕过。

## NPU端运行dense

先用 `npu-smi info` 选空闲卡。以下物理卡4映射为逻辑0；`192.0.2.10`需替换为GPU服务器实际地址，端口需与服务或容器映射端口一致：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 atk node \
  --backend npu --name dut --devices 0 \
  --output_path /srv/fa4-results/online-dense-001 \
  node \
  --backend gpu --name same --devices 0 \
  --host 192.0.2.10 --port 9090 \
  task \
  --case /srv/fa4-precision/cases/all_flash_attn_func.json \
  --task accuracy --bm_device cpu --syc_dataset --timeout 3600 \
  --plugin_path /srv/fa4-precision/executor_flash_attn_fa4.py \
  --save_report_json --save_data input:bin --save_data output:bin
```

## NPU端运行varlen

```bash
ASCEND_RT_VISIBLE_DEVICES=4 atk node \
  --backend npu --name dut --devices 0 \
  --output_path /srv/fa4-results/online-varlen-001 \
  node \
  --backend gpu --name same --devices 0 \
  --host 192.0.2.10 --port 9090 \
  task \
  --case /srv/fa4-precision/cases/all_flash_attn_varlen_func.json \
  --task accuracy --bm_device cpu --syc_dataset --timeout 3600 \
  --plugin_path /srv/fa4-precision/executor_flash_attn_fa4.py \
  --save_report_json --save_data input:bin --save_data output:bin
```

- `--syc_dataset` 使用主节点统一生成的输入同步到远端，不能只依赖相同seed。
- `--bm_device cpu` 自动创建高精度benchmark任务，不再添加普通CPU node。
- NPU节点在前；这里GPU是在线节点，不使用 `accuracy_load` 或 `bm_output_path`。
- 首次建议在task部分加 `-s 0 -e 1` 单例穿刺；varlen显式SplitKV代表例为 `-s 14 -e 15`，结束位置不包含。全量移除切片，两组串行运行，结果目录每次换新。
- 原 `result/` 是历史单标杆源资产，请使用现成 `cases/`；`run_precision.py` 仅保留为离线可选工具，不是在线验收前提。

## GPU硬件边界

核对的官方FA4 revision `10b74d98fb26113b014b33d947c8bbe4e83fbf7a` 中，A100（SM80）和H20（SM90）原生不支持显式 `num_splits=2/4`。按验收允许的“相同数学语义、不要求相同split调度”策略，**GPU精度标杆现在固定 `num_splits=1`（原用例0/1/2/4均如此），NPU仍使用原case参数，CPU仍计算FP64真值**。保留全部200例，包含27个varlen显式SplitKV case，不改JSON、不删例、不切换FA2/FA3。该适配消除GPU SplitKV分派阻塞，但不是GPU/NPU性能比较，也不证明设备上的其他FA4兼容条件均已验证。

部署更新后的精度目录后，重启GPU端 `atk server` 使其重新加载标杆。上面的裸ATK命令无需更改。离线可选工具需重新prepare和采集参考数据，旧manifest/输出不能复用。

完整部署、生成用例、参考语义和验收限制见[精度工程README](tests/atk/atk-precision/README.md)。本次未执行GPU/NPU或真实联机验证；以正式运行的逐例状态及主输出、LSE双标杆报告为准，不以退出码0代替验收。

# atk性能用例执行（FA3/FA4，950，裸ATK命令）

性能工程为 `tests/atk/atk-perf`，与上面的双标杆精度工程完全分开。dense、varlen各50个固定case（ID 0–49），FA3/FA4共用相同JSON；**性能采集仅需NPU，不需要GPU server**。性能执行器中的CPU节点仅返回占位结果，不是精度真值，其性能比值和是否达标不能用来判断FA3/FA4加速比。

## 准备与单组执行模板

先在仓库根按正常流程构建FA3/FA4扩展，加载匹配的CANN/torch_npu环境，并用 `npu-smi info` 选择空闲950卡。下面以物理卡4为例，ATK使用映射后的逻辑0。请将 `PERF` 改为本机性能目录的绝对路径；当前工作区可使用：

```bash
PERF=/home/jisihuai/working/flash-attention-npu/tests/atk/atk-perf
export ASCEND_RT_VISIBLE_DEVICES=4
```

设置一组测试变量后直接执行裸ATK命令：

```bash
VERSION=fa4                         # fa3 或 fa4
API=flash_attn_func                  # 或 flash_attn_varlen_func
TASK=performance_e2e                # 或 performance_device / performance_device_pta

FLASH_ATTN_PERF_VERSION="$VERSION" atk node \
  --name npu_dut --backend npu --devices 0 \
  --output_path "$PERF/atk_output/$VERSION/$API/$TASK" \
  node --name cpu_golden --backend cpu \
  --output_path "$PERF/atk_output/$VERSION/$API/$TASK" \
  task --case "$PERF/$API.json" --task "$TASK" \
  --performance_data 100,50,40 \
  --plugin_path "$PERF/executor_fa_perf.py" \
  --single_process --max_task 1 --timeout 3600 --need_time false
```

- `FLASH_ATTN_PERF_VERSION` 必须明确指定；未设置时执行器默认FA4。
- `100,50,40` 分别为预热次数、性能采集次数、取后N次平均。
- E2E测执行器/API路径，包含host开销；Device测profiler统计的设备耗时；PTA观察分配内存，不是耗时或独立峰值workspace估计。
- 初次可在task部分增加 `-s 0 -e 1` 只跑第一例，移除后才是50例全量。
- 两版本使用同一张卡和相同参数，所有任务串行执行，不要将两个命令后台并发。**复跑前确认输出目录未使用**；`--need_time false`及下面的汇总约定使用固定目录，不能混入旧报告。要保留多轮，可另建完整工程副本并修改 `PERF`，不要覆盖原报告。

## 三类指标全量串行采集

若要采齐FA3/FA4 × 两接口 × 三指标，可以直接执行下列shell循环；每轮调用的仍是裸 `atk`，不依赖 `run_perf.sh`：

```bash
for VERSION in fa3 fa4; do
  for API in flash_attn_func flash_attn_varlen_func; do
    for TASK in performance_e2e performance_device performance_device_pta; do
      FLASH_ATTN_PERF_VERSION="$VERSION" atk node \
        --name npu_dut --backend npu --devices 0 \
        --output_path "$PERF/atk_output/$VERSION/$API/$TASK" \
        node --name cpu_golden --backend cpu \
        --output_path "$PERF/atk_output/$VERSION/$API/$TASK" \
        task --case "$PERF/$API.json" --task "$TASK" \
        --performance_data 100,50,40 \
        --plugin_path "$PERF/executor_fa_perf.py" \
        --single_process --max_task 1 --timeout 3600 --need_time false || break 3
    done
  done
 done
```

每组50例，共12组、600个case-task组合。如果只需要E2E，可将任务循环列表改为仅 `performance_e2e`。`break 3`只在进程非零退出时停止；**ATK退出码0仍可能有失败case**，不得仅凭循环结束认定完整成功，须检查每组日志/报告的NPU运行状态和有效指标，异常时人工停止后续采集。

按现有目录约定汇总：

```bash
python "$PERF/summarize_results.py" \
  --root "$PERF" --output "$PERF/results"
```

输出 `results/fa3_fa4_performance_comparison.csv`、`results/fa3_fa4_performance_summary.csv`。汇总器按API/case ID配对，计算 **FA3耗时/FA4耗时**，大于1表示FA4更快；缺失/非有限/非正耗时不参与比值。它会寻找每组最新XLSX，不会证明多个历史批次具有相同环境；应先保证目录干净、来源一致。重复汇总会覆盖同名CSV，需要保留时改用新的 `--output`。

## Device profile与FAInfer单kernel采集

如果还需保存原始profile并分别比较ATK Device汇总耗时、FAInfer单kernel耗时，使用独立目录。以下四组裸ATK串行采集，不写入上面的三指标结果树：

```bash
PROFILE_ROOT="$PERF/device_profile_full_run001"  # 必须是新的结果目录
for VERSION in fa3 fa4; do
  for API in flash_attn_func flash_attn_varlen_func; do
    FLASH_ATTN_PERF_VERSION="$VERSION" atk node \
      --name npu_dut --backend npu --devices 0 \
      --output_path "$PROFILE_ROOT/atk_output/$VERSION/$API" \
      node --name cpu_golden --backend cpu \
      --output_path "$PROFILE_ROOT/atk_output/$VERSION/$API" \
      task --case "$PERF/$API.json" --task performance_device \
      --performance_data 100,50,40 --save_data profile \
      --plugin_path "$PERF/executor_fa_perf.py" \
      --single_process --max_task 1 --timeout 3600 --need_time false || break 2
  done
 done
```

每组结束后检查50个唯一case ID、NPU状态和50个正数Device耗时，以及profile是否齐全；缺失或计时为0不能当作有效结果。四组全部完成后：

```bash
python "$PERF/summarize_device_profiles.py" --root "$PROFILE_ROOT"
```

输出在 `$PROFILE_ROOT/results/`：

- `fa3_fa4_device_profile_comparison.csv`：逐case的两类耗时和加速比。
- `fa3_fa4_device_profile_summary.csv`：按API汇总。
- `fa3_fa4_device_profile_kernel_samples.csv`：FAInfer逐调用明细。
- `run_metadata.json`：采集参数、报告映射等信息。

FAInfer从kernel名称包含 `FAInfer` 的正数duration提取，每case预期50次调用，取**中位数，不把调用相加**。ATK Device汇总值不等于单条FAInfer耗时；profile开启有采集开销，不与无profile绝对耗时混为一谈。出现 `op_summary` 缺失/零耗时应排查采集环境，不能填零计算加速比。

现有 `run_perf.sh`、`run_device_profile_full.sh` 仍可选使用；后者额外按组检查50个正数Device耗时且要求物理卡4，但不是裸命令验收的前提。重装前README及恢复CSV中的历史结果不代表当前环境已复现。**本节只整理命令，未执行性能采集，也未修改性能工程。**
