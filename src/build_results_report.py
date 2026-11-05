"""Build a Chinese reproducibility report from completed saved experiments.

No expected physical result is substituted for an observation. Raw data and
numeric results are read-only. Output is a human-readable Markdown report and
an explicit machine-readable evidence summary, not the original project DOCX.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np

from evidence import small_scale_mask
from run_analysis import save_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def mean_signal(path, limits):
    a = arrays(path)
    mask = small_scale_mask(a["edges"], limits) & np.isfinite(a["xi"])
    mask &= np.isfinite(a["covariance"]).diagonal()
    if not mask.any():
        return {"mean": None, "internal_jk_se": None, "bins": []}
    weights = np.full(int(mask.sum()), 1/mask.sum())
    variance = weights @ a["covariance"][np.ix_(mask, mask)] @ weights
    return {"mean": float(np.mean(a["xi"][mask])), "internal_jk_se": float(np.sqrt(max(0, variance))),
            "bins": np.flatnonzero(mask).tolist()}


def fmt(x, digits=3):
    return "不可用" if x is None else f"{x:.{digits}f}"


def main():
    cfg = load(ROOT/"config.json")
    science = ROOT/"results/science"
    manifest = load(science/"run_manifest.json")
    if load(science/"run_state.json")["status"] != "all_12_cases_computations_complete":
        raise RuntimeError("All twelve cases must finish before building the final report")
    tests = load(ROOT/"results/verification.json")
    records = []
    for snap in cfg["snapshots"]:
        name = f"snapshot_{snap:03d}"
        records.append({"snapshot": snap, "name": name,
            "quality": load(ROOT/"prepared"/(name+".quality.json")),
            "summary": load(science/name/"summary.json"),
            "audit": load(science/name/"evidence_audit.json"),
            "signal": mean_signal(science/name/"global.npz", cfg["small_scale_range_cMpc_h"]),
            "control_summary": load(science/(name+"_poisson_control")/"summary.json"),
            "control_audit": load(science/(name+"_poisson_control")/"evidence_audit.json"),
            "control_signal": mean_signal(science/(name+"_poisson_control")/"global.npz", cfg["small_scale_range_cMpc_h"]),
            "matched_summary": load(science/(name+"_mass_matched")/"summary.json"),
            "matched_audit": load(science/(name+"_mass_matched")/"evidence_audit.json"),
            "matched_signal": mean_signal(science/(name+"_mass_matched")/"global.npz", cfg["small_scale_range_cMpc_h"]),
            "benchmark": load(ROOT/"results"/f"benchmark_{snap:03d}"/"benchmark_summary.json")})
    engineering = load(ROOT/"results/benchmark_engineering_100k/benchmark_summary.json")
    engineering_source = load(ROOT/"benchmark_catalogues/snapshot_099_scaling_100k.engineering-only.provenance.json")
    null_ensemble = load(ROOT/"results/uniform_null_ensemble/summary.json")
    null_pair_check = load(ROOT/"results/uniform_null_ensemble/original_033_pair_check.json")
    matching = load(science/"mass_matching.json")
    differences = load(science/"cross_snapshot_differences.json")
    all_bins = np.asarray(cfg["edges_cMpc_h"])
    small = small_scale_mask(all_bins, cfg["small_scale_range_cMpc_h"])
    actual_small = [float(all_bins[:-1][small].min()), float(all_bins[1:][small].max())]
    environment = {"python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "packages": {p: metadata.version(p) for p in
         ["numpy", "scipy", "pandas", "matplotlib", "h5py", "hdbscan", "scikit-learn", "pytest", "psutil", "seaborn", "pypdf"]},
        "environment_source": "New GalaxyWork venv from bundled Python; third-party site-packages copied from the existing validated project environment; not a fresh online install",
        "base_runtime": sys.base_prefix, "source_site_packages_copy": "/Users/wanningkang/Documents/Codex/2026-08-07/zhe/work/galaxy-env/lib/python3.12/site-packages",
        "thread_limits": {"OPENBLAS_NUM_THREADS": 1, "OMP_NUM_THREADS": 1},
        "notes": "No credentials or unrelated environment variables recorded"}
    save_json(ROOT/"results/environment.json", environment)
    save_json(ROOT/"results/evidence_summary.json", {"records": records,
        "engineering_benchmark": engineering, "engineering_provenance": engineering_source,
        "actual_small_scale_edges_cMpc_h": actual_small, "matching": matching,
        "paired_cross_snapshot_differences": differences, "verification": tests,
        "uniform_null_ensemble": null_ensemble, "original_033_independent_pair_check": null_pair_check})

    lines = ["# TNG 星系空间分析实验过程与结果", "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}。数据为官方 TNG100-1 模拟的四快照目录，不是观测巡天星系，也不是本项目自行生成的星系模型；均匀随机对照另行明确标注。", "",
        "## 本次完成的实验", "",
        "本次完成四个快照的预处理、真实数据性能测试、全局和局部分析、四种联合分析，以及四套同数量均匀随机对照和四套跨红移恒星质量直方图匹配分析，共十二套科学计算。原始下载数据和原 Desktop 项目均未修改。本报告由保存的数值结果生成，文档里的预计发现没有被当作观测结果。", "",
        "`README.md` 是操作入口；`src/` 保存代码，`tests/` 保存测试，`prepared/` 保存标准化点表与质量报告，`results/` 保存逐案例输出、日志、清单和图。运行不需要再输入 API key。", "",
        "## 数据来源与质量控制", "",
        "真实数据输入为 `/Users/wanningkang/GalaxyWork/data/raw/TNG100-1/`。每个快照读取五个字段：SubhaloPos、SubhaloMassType、SubhaloLenType、SubhaloFlag、SubhaloGrNr。每个 HDF5 的形状、行数、来源快照元数据和本地下载清单的 SHA256/字节数均已核对。字段子集文件本身没有 snapshot Header，因此不能声称仅凭文件内部 Header 独立证明了快照身份；来源依赖 API 路径和下载时保存的元数据/清单。本地清单校验也不是上游独立发布的 checksum 验证。", "",
        "正式筛选固定为 SubhaloFlag=1、总 bound stellar mass≥1e9 Msun、恒星粒子数≥100；恒星质量使用 SubhaloMassType 的第 4 索引分量乘 1e10/h，h=0.6774。坐标只从 ckpc/h 除以1000换为 cMpc/h，保持边长75 cMpc/h的周期盒，不做 z-score，也不额外除以h。上述阈值是项目选择，并非官方完备性保证。", "",
        "粒子数阈值具体使用目录字段SubhaloLenType[:,4]，是Type4数量的分辨率代理。本次没有下载粒子级形成时间以逐个区分真实恒星与wind-phase对象，因此不能将其宣传成独立验证的纯恒星粒子计数。", "",
        "| Snapshot | 实际红移 | 原始 subhalo 数 | 正式入选星系数 | 入选重复坐标/ID |",
        "|---|---:|---:|---:|---:|"]
    for r in records:
        q = r["quality"]
        lines.append(f"| {r['snapshot']} | {q['redshift']:.5g} | {q['n_raw']:,} | {q['n_selected']:,} | 0 / 0 |")
    lines += ["", "Snapshot 99 有一条原始越界记录：Subfind ID 825117 的 y=75000.0390625 ckpc/h。其恒星质量约7.54e6 Msun、恒星粒子数7，独立不满足正式筛选，故无需修正坐标即可排除。质量报告保留原始值；符合正式筛选条件的对象若越界仍会触发停止。恰好等于盒长的坐标分量仅按周期等价映射为0。没有把任意越界对象静默 modulo 回盒内。", "",
        "每个快照的 `*.quality.json` 保存逐条件的非互斥排除数；这些数字不能相加作为总删除数。同坐标不同ID不预设为错误；本次入选目录没有此类重复。Subfind ID仅在一个快照内标识对象，跨快照并非同一星系的持久ID。", "",
        "## 代码验证与性能测试", "",
        f"自动测试结果：{tests['pytest_summary']}；另外工程目录生成脚本的内存自测通过。数值代码增加了同随机目录的JK-RR缓存、真实异常记录审计、不同实际图k的稳健性检查和JK参数区间有效性检查，修改只发生在GalaxyWork副本。", "",
        "周期最小像距离、分箱端点、无序自配对、交叉配对和规范化规则在暴力法与cKDTree间保持一致。正式样本每快照测试N=500、1000、5000、10000，各3次重复；分块暴力上限5000，完整距离矩阵基线上限1000。四快照共60次方法比较、180个DD/DR/RR直方图完全一致；smoke另5次比较通过。超过入选样本数的50K/100K测试明确跳过。", "",
        "| Snapshot | N=5000 cKDTree中位数 秒 | 分块暴力中位数 秒 | 加速比 |",
        "|---|---:|---:|---:|"]
    for r in records:
        b = r["benchmark"]
        metrics = {v["method"]: v for v in b["metrics"] if v["n"] == 5000}
        tree, brute = metrics["ckdtree"]["elapsed_total_s"]["median"], metrics["blocked_brute"]["elapsed_total_s"]["median"]
        lines.append(f"| {r['snapshot']} | {tree:.5f} | {brute:.5f} | {brute/tree:.2f}× |")
    tree100 = next(v for v in engineering["metrics"] if v["method"] == "ckdtree" and v["n"] == 100000)
    lines += ["", "### 独立的100K工程扩展性测试", "",
        f"为核验大规模计算而不改变正式物理样本，另从Snapshot99中Flag=1、恒星质量>0、恒星粒子数≥10且坐标合法的{engineering_source['audit']['n_candidates']:,}个对象，无放回抽取100,000个真实对象。它是明确标注的 engineering-only 样本，不是≥1e9 Msun且粒子数≥100的物理主样本。五字段来源、候选ID与抽样ID哈希、NPZ哈希和选择规则保存在 `benchmark_catalogues/`。没有生成、复制或抖动星系来凑数量。", "",
        f"工程测试覆盖500到100,000点，每规模3次重复；100,000点cKDTree的DD+DR+RR总时间中位数为{tree100['elapsed_total_s']['median']:.4f}秒，whole-worker峰值RSS中位数{tree100['peak_rss_bytes']['median']/1048576:.2f}MiB。100K暴力法没有运行，因此没有100K暴力加速比，也不能宣称在100K直接验证了点对完全一致。", "",
        "计时包括两棵树建树和DD/DR/RR查询，不包括输入读取、抽样及导入；RSS包括整个独立worker的解释器、导入、输入数组和计算高水位，不是纯算法增量内存。分块暴力内存受block=512约束，不能把它说成始终需要N²内存。速度数字仅对应本机、当前分箱与样本。", "",
        "## 全局聚集与幂律拟合", "",
        "同一套规范化DD/DR/RR用于Natural、Davis–Peebles、Hewett、Hamilton、Landy–Szalay五个估计器；后者是主结果。随机目录为正式样本的5倍，另检查随机种子和目录规模。拟合的是体积加权距离壳平均幂律，而非把每箱当作中心点的精确值。", "",
        f"预先声明的小尺度区间是0.2–1 cMpc/h；当前只汇总完整落在其中的3个bin，实际覆盖{actual_small[0]:.6f}–{actual_small[1]:.6f} cMpc/h。下表小尺度ξ为这些bin的等权平均，SE来自完整协方差，不能当作整个0.2–1区间的积分。", "",
        "| Snapshot | 小尺度平均ξ ± 内部JK SE | r0 cMpc/h | gamma | reduced χ² |",
        "|---|---:|---:|---:|---:|"]
    for r in records:
        s, v = r["summary"], r["signal"]
        fit = s["fit"]
        lines.append(f"| {r['snapshot']} | {fmt(v['mean'])} ± {fmt(v['internal_jk_se'])} | {fmt(fit.get('r0'))} | {fmt(fit.get('gamma'))} | {fmt(fit.get('reduced_chi_square'))} |")
    lines += ["", "r0/gamma只是在预设拟合尺度的参数化概括。成功收敛不意味着单一幂律充分；需结合上表reduced χ²、残差图、协方差正则化及边界命中信息。每个summary保留27次delete-one拟合；只有拟合bin完全相同、均无参数边界命中且参数有限时才给出近似正态JK参数区间。", "",
        "空间jackknife将周期盒分为27区，每次对D和R删除同一区域并保留原盒长。它估计内部空间不确定性，不能包含盒尺度以上宇宙方差；固定标签子样本的误差不包括重新发现标签的不确定性。没有把不同随机种子的波动或随机删除的波动称作宇宙方差。", "",
        "## 局部结构与图近似检查", "",
        "在同一套正式入选样本上，用第20个其他近邻距离构造k/V局部数量密度，另用k=10检查排序。高/中/低密度比例为20/60/20。HDBSCAN在原始三维周期几何上识别数据驱动群体，不把群体自动当作物理星系团。", "",
        "| Snapshot | HDBSCAN群体数 | noise比例 | k20/k10密度秩相关 | 全样本图32/64的ARI | 小样本稠密/稀疏ARI |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in records:
        s, a = r["summary"], r["audit"]
        graph = a["hdbscan_sensitivity"]["graph_validation"]
        lines.append(f"| {r['snapshot']} | {s['n_clusters']} | {s['noise_fraction']:.2%} | {s['density_rank_spearman']:.4f} | {fmt(graph.get('full_graph_sensitivity_ari'),4)} | {fmt(graph.get('ari_including_noise'),4)} |")
    lines += ["", "大样本使用稀疏周期邻居图近似，完整kNN图连通并不证明它恢复了完整距离矩阵的MST。实际k、小样本500点的稠密/稀疏比较、更高不同实际k、min_cluster_size=20/40/80敏感性均保留。ARI包含noise标签；很高的ARI不能消除所有近似误差。noise表示算法未归类，不代表文件错误；membership strength也不是物理成员概率。改变min_cluster_size后的标签并不完全相同，不能称所有群体对参数完全稳定。", "",
        "群体中位密度与以下描述量的Spearman秩相关：", "",
        "| Snapshot | 紧凑度 | persistence | 平均membership strength |",
        "|---|---:|---:|---:|"]
    for r in records:
        relations = r["audit"]["cluster_relations_to_median_density"]
        lines.append(f"| {r['snapshot']} | {fmt(relations['compactness_descriptor']['spearman'])} | {fmt(relations['cluster_persistence']['spearman'])} | {fmt(relations['mean_membership_strength']['spearman'])} |")
    lines += ["", "本次密度与紧凑度/persistence正相关，但与平均membership strength负相关，不能照搬文档中“密度越高平均成员强度越高”的预计说法。上述指标共享位置和聚类构造，相关也不是独立物理验证。", "",
        "## 四种全局与局部联合分析", "",
        "第一项比较高低密度子样本，第二项比较cluster/noise，第三项比较定向删除与同数量随机删除，第四项分解原始DD。子样本重新计算DD/DR并用新ND规范化；同随机目录的RR才复用。", "",
        "| Snapshot | 高密度减低密度ξ ± JK SE | cluster减noise ξ ± JK SE | 删除最高10%后平均ξ变化 | 同群体原始DD占比 |",
        "|---|---:|---:|---:|---:|"]
    for r in records:
        a = r["audit"]
        high, cluster = a["joint_1_density_strata"], a["joint_2_cluster_noise"]
        removal = a["joint_3_density_removal"]["10"]
        share = a["joint_4_raw_DD_sources"]["within"]["raw_pair_share"]
        relative = removal['targeted_mean_delta_xi']/r['signal']['mean']*100
        lines.append(f"| {r['snapshot']} | {fmt(high.get('mean_difference'))} ± {fmt(high.get('internal_jk_se'))} | {fmt(cluster.get('mean_difference'))} ± {fmt(cluster.get('internal_jk_se'))} | {fmt(removal['targeted_mean_delta_xi'])}（{relative:.1f}%） | {share:.2%} |")
    lines += ["", "删除列的百分比是三个bin等权均值的相对变化，不是逐bin百分比的平均。上述联合结果是描述性关联：密度和群体标签均由同一坐标生成，按标签选择本身就可能产生相关函数差异。总DD非零时，原始DD四类占比精确相加为1，但不是Landy–Szalay超额相关信号的可加贡献率。同群体小尺度DD占比较高，不能直接写成该群体因果驱动了相同比例的全局信号。", "",
        "### 同数量均匀随机对照", "",
        "以下对照采用相同N、周期盒和全套分析流程。它是合成均匀坐标，继承的真实source SHA仅指明母目录的N与对象属性来源，不是合成坐标的独立真实性证明；最终输出文件另有完整性哈希。", "",
        "| Snapshot对应N | 真实样本平均ξ | 均匀对照平均ξ ± JK SE | 均匀对照高密度减低密度ξ | 均匀对照群体数 |",
        "|---|---:|---:|---:|---:|"]
    for r in records:
        v, c, a = r["signal"], r["control_signal"], r["control_audit"]
        lines.append(f"| {r['snapshot']} | {fmt(v['mean'])} | {fmt(c['mean'])} ± {fmt(c['internal_jk_se'])} | {fmt(a['joint_1_density_strata'].get('mean_difference'))} | {r['control_summary']['n_clusters']} |")
    lines += ["", "均匀对照也可能产生HDBSCAN标签和高低密度分组差异，说明这些标签本身不能独立证明物理群体。真实样本与对照的量级及曲线应结合阅读，不将对照标签解释为真实星系团。", "",
        "### 发现的零信号对照偏离与追加检查", "",
        "原始z≈2均匀对照的小尺度均值为−0.322，内部JK SE约0.081，不能说四个单次对照均与0一致。没有删除或更换这个不利结果。为检查有限样本/有限随机目录的波动，追加固定20种子、每种子独立生成D与R的全局分析（每个真实N均做20次）；设计和所有结果保存在 `uniform_null_ensemble/`。", "",
        "| Snapshot对应N | 20次均值 | 实现间SD | 原始单次值距追加均值的SD数 |",
        "|---|---:|---:|---:|"]
    for item in null_ensemble:
        lines.append(f"| {item['snapshot']} | {fmt(item['mean_small_scale_xi'],4)} | {fmt(item['between_realisation_sd'],4)} | {fmt(item['original_distance_from_ensemble_mean_in_sd'],2)} |")
    lines += ["", "20次追加均值在−0.035到0.030之间，相对真实样本均值45.8–141.8小很多；但原始z≈2负偏离仍位于这个小型集合的尾部，20次也不足以校准尾部概率或证明只是偶然。一些20次均值的近似均值区间本身不含0，因此本报告不宣称严格零信号/误差校准检验全部通过。", "",
        "还用短距离边枚举（query_pairs及稀疏交叉边）加显式周期距离，独立复核原始z≈2对照三个bin的DD/DR/RR，全部与count_neighbors结果一致。它排除了这些bin的点对算术不一致，不能消除有限实现偏离或证明JK误差模型完全校准。这是本次不确定性验证保留的限制。", "",
        "## 跨红移比较", "",
        "四个快照来自同一个模拟盒和共同初始条件，既不能混合成一个all样本，也不能当四个独立宇宙体积。除了相同质量阈值的主分析，按固定log10恒星质量区间取四快照共同直方图计数，形成相同N的附加样本；粗bin匹配不能完全匹配bin内质量或保证完备性。", "",
        "| Snapshot | 质量匹配N | 匹配后小尺度平均ξ ± 内部JK SE | 匹配后r0 |",
        "|---|---:|---:|---:|"]
    for r in records:
        v, s = r["matched_signal"], r["matched_summary"]
        lines.append(f"| {r['snapshot']} | {s['n']:,} | {fmt(v['mean'])} ± {fmt(v['internal_jk_se'])} | {fmt(s['fit'].get('r0'))} |")
    lines += ["", "跨红移差值及匹配空间区域的协方差保存在 `cross_snapshot_differences.json`，主样本和质量匹配样本都有记录。不能只看r0大小或误差条重叠就宣布红移演化显著，也不能提前保证低红移聚集更强。", "",
        "## 结论与简历可使用的表述", "",
        "计算层面已获得实证：在真实小样本上三类点对计数完全一致，5K规模约25倍加速；另以明确分开的真实工程目录完成100K扩展性测试。正式物理目录最多21,052个对象，因此不能把100K工程测试说成100K高质量主样本的科学分析。", "",
        f"科学层面实际观测到：三个预定义bin的均值从z≈2的{records[-1]['signal']['mean']:.2f}增加到z≈0的{records[0]['signal']['mean']:.2f}；粗质量直方图匹配后，z≈2/z≈0仍分别为{records[-1]['matched_signal']['mean']:.2f}/{records[0]['matched_signal']['mean']:.2f}。删除最高密度10%的星系使该汇总均值下降约31.7%–62.8%，同数量随机删除的均值变化小得多。同群体内点对占该尺度原始DD约72.2%–89.6%。这些结果支持“本次选择下，小尺度聚集与局部高密度、紧凑群体相关，且低红移聚集更强”的描述。", "",
        "该联系不是独立因果证据，不能把原始DD占比等同于超额信号的贡献率。r0并不严格单调，幂律拟合不足、聚类参数依赖、坐标选择效应、单次零对照偏离和内部误差未完全校准的局限必须保留。", "",
        "可采用的英文简历表述：", "",
        "- Preprocessed TNG100-1 subhalo catalogues across four redshift snapshots, selecting 12,281–21,052 galaxy-hosting subhalos per snapshot with consistent stellar-mass and resolution cuts, comoving units, periodic geometry, and auditable quality checks.",
        "- Validated periodic cKDTree DD/DR/RR counts against brute-force references on 500–5,000-point samples, achieving an approximately 25× pair-count speedup at 5,000 points; separately benchmarked scaling to 100,000 real TNG objects using an explicitly broader engineering-only sample.",
        "- Combined kNN local density, periodic sparse-graph HDBSCAN, and two-point correlation estimates to compare density environments, cluster/noise subsets, and the spatial composition of small-scale galaxy pairs.",
        "- Evaluated five correlation estimators, nonlinear power-law fit diagnostics, 27-region spatial jackknife uncertainty, graph/parameter sensitivity, density-removal controls, and mass-matched cross-redshift comparisons to assess descriptive global–local clustering relationships.", "",
        "不建议写“证明高密度群体因果驱动聚集”“100K暴力法完全一致”“单一幂律在所有尺度拟合良好”或“低红移必然更强”。这些都超出本次实验能支持的范围。", "",
        "## 复现和审计文件", "",
        "`src/run_staged.py` 是科学分析分阶段入口；`src/run_benchmarks.py` 是正式样本性能入口；`src/prepare_scaling_catalog.py` 单独产生100K工程输入。`config.json`固定参数；`results/environment.json`记录实际环境；`results/verification.json`记录测试与最终完整性核验。", "",
        "每个case有run manifest、源NPZ哈希、数值代码指纹、完整日志、case_complete标记和evidence_audit。运行时只核对完成标记的断点逻辑，在交付前另做全文件存在性、NPZ可读性、关键数值不变量和输出SHA256核验。未把无Header子集文件的来源核验说成独立内容鉴定。", "",
        "## 核心图表入口", "",
        f"- [质量匹配后的跨红移曲线]({ROOT}/results/joint_figures/cross_redshift_mass_matched.png)",
        f"- [Snapshot99密度分组曲线]({ROOT}/results/figures/snapshot_099_density_subsets.png)",
        f"- [Snapshot99定向删除与随机删除]({ROOT}/results/joint_figures/snapshot_099_removal.png)",
        f"- [100K工程性能曲线]({ROOT}/results/engineering_figures/benchmark_time.png)",
        f"- [保留原始异常对照的20种子追加检查]({ROOT}/results/joint_figures/uniform_null_ensemble.png)",
        "", "共41张图，均另有PDF、SVG、PNG和灰度版；图表标注了分箱、显示抽样、误差类型和局限。图面/格式检查不能代替科学误差校准。", "",
        "## 定义与参考", "",
        "- [TNG字段与单位](https://www.tng-project.org/data/docs/specifications/)：SubhaloPos、SubhaloMassType、SubhaloLenType、SubhaloFlag、SubhaloGrNr。",
        "- [TNG宇宙学、模拟背景与引用要求](https://www.tng-project.org/data/docs/background/)：h=0.6774，正式发表需按官方要求引用相应论文。",
        "- [SciPy cKDTree点对计数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.count_neighbors.html)：非累计距离分箱和端点定义。", ""]
    (ROOT/"实验过程与结果.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {ROOT/'实验过程与结果.md'}")


if __name__ == "__main__":
    main()
