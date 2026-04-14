# run_dual_source 数据构造与划分审计（LOTSA / GiftEval）

## 1) 多通道如何构造成单通道

### 结论
- 当前代码路径是 **“强制单通道 + 多通道按通道拆成多条独立序列”**，不是“只取目标通道”。

### 证据链
1. `generate_correction_dataset.py` 在加载原始数据时固定 `force_univariate=True`。
2. `utils/data.py` 的 `Dataset.gluonts_dataset` 在 `force_univariate=True` 分支中：
   - 若 `target.ndim > 1`，则按 `for i in range(C)` 把每个通道拆成一条新序列；
   - 并把 `item_id` 改为 `"{base_id}_dim{i}"`。

因此，LOTSA 和 GiftEval 的多通道样本会被拆成多条单通道样本进入后续窗口生成。

---

## 2) sequence 划分 vs 时间前后划分：潜在漏洞

## 2.1 `seq_per_dataset`（按 sequence 划分）

### 关键实现
- `generate_correction_dataset.py` 中 `sample_metas` 的 `seq_id` 是 `enumerate(dataset_obj.gluonts_dataset)` 产生的索引。
- 因为上游已经把多通道拆成多条序列，所以 `seq_id` 已经是“拆分后的通道序列 ID”。
- `corrector/trainer.py` 的 `seq_per_dataset` 以 `seq_id` 为单位 80/20 随机分配 train/test。

### 漏洞
- 来自同一个原始多通道序列的不同通道（例如 `base_id_dim0` 和 `base_id_dim1`）可能被分到 train/test 两边。
- 若跨通道相关性高，这会造成“近重复信息泄漏”，让 `seq_per_dataset` 看起来异常好。

## 2.2 `temporal_per_seq`（同序列前80%后20%）

### 关键实现
- 仍按 `seq_id` 分组，但在每个 `seq_id` 内按时间窗口排序后切 80/20。

### 漏洞
- 同一通道序列前后窗口高度自相关，训练集和测试集共享同一底层序列，只是时间段不同。
- 在平稳序列/强季节序列上，任务难度可能明显降低。

---

## 3) 为什么会出现“按 sequence 划分反而更好”

可能原因（按当前实现）：
1. `seq_per_dataset` 并非“按原始多变量序列分组”，而是“按拆分后的单通道分组”。
2. 同源多通道可跨 train/test，带来隐性泄漏，导致结果偏乐观。
3. `seq_per_dataset` 分支还取消了 train/test group 的硬限制（注释写明），会改变预期实验协议。

---

## 4) 建议修复方向

1. 在 `utils/data.py` 拆通道时保留 `parent_item_id`（不含 `_dimX`）。
2. 在 `generate_sliding_windows` 里把 `parent_item_id` 写入 `sample_metas`。
3. 在 `trainer.py` 的 `seq_per_dataset` / `temporal_per_seq` 增加 `group_by_parent_item_id` 选项：
   - 若启用，先按 `parent_item_id` 分组再划分，避免同源通道跨集。
4. 对 GiftEval / LOTSA 额外输出“同源跨集率”审计指标（train/test 中共享 parent 的比例）。

