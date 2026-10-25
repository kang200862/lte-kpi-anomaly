# -*- coding: utf-8 -*-
"""
kpi_anomaly_analysis.py —— LTE 小区 KPI 异常检测（业务规则法，单文件版）

一条命令完成：读入导出数据 -> 标准化清洗 -> 三条业务规则判定 -> 输出标准工单表。
规则口径：
  R1 绝对阈值：接入成功率 < 95%，掉话率 > 2%，切换成功率 < 92%          -> 重要
  R2 波动：单小时指标低于该小区当日均值的 70%（掉话率方向相反，高于 130%） -> 一般
  R3 持续故障：同一小区连续 >= 3 小时触发 R1                            -> 紧急
用法：python kpi_anomaly_analysis.py [输入CSV]
      不传参数时默认读 data/clean_lte_kpi.csv，工单写入 data/anomaly_work_order.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== 配置区 ==============================
BASE_DIR = Path(__file__).resolve().parent          # 项目根目录（脚本所在目录）
DEFAULT_INPUT = BASE_DIR / "data" / "clean_lte_kpi.csv"   # 默认输入
OUT_FILE = BASE_DIR / "data" / "anomaly_work_order.csv"   # 工单输出
ENCODING = "utf-8-sig"        # 带 BOM，Excel 双击打开中文不乱码
TIME_FMT = "%Y-%m-%d %H:%M"   # 统一后的时间格式

# 原始导出列名 -> 标准列名（换数据源时只改这张表）
RENAME_MAP = {
    "接入成功率(%)": "接入成功率",
    "掉话率(%)": "掉话率",
    "下行吞吐量(Mbps)": "下行吞吐量",
    "切换成功率(%)": "切换成功率",
}

# 清洗后保留的核心字段（顺序即输出顺序）
CORE_COLUMNS = ["小区ID", "时间", "接入成功率", "掉话率", "下行吞吐量", "切换成功率"]

# 参与检测的数值指标
VALUE_COLUMNS = ["接入成功率", "掉话率", "下行吞吐量", "切换成功率"]

# R1 绝对阈值：(指标, 方向, 门限)  lt=低于算异常  gt=高于算异常
ABS_RULES = [("接入成功率", "lt", 95.0), ("掉话率", "gt", 2.0), ("切换成功率", "lt", 92.0)]

WAVE_RATIO = 0.70          # R2：低于当日均值的 70%
WAVE_RATIO_UPPER = 1.30    # R2：掉话率这类越高越差的指标，高于当日均值的 130%
MIN_HOURS_PER_DAY = 6      # 当天有效小时数少于此值，不参与波动判定
PERSIST_HOURS = 3          # R3：连续多少小时触发 R1 判为持续故障

# 工单字段（固定 7 列，顺序不可变）
WORK_ORDER_COLUMNS = ["小区ID", "异常开始时间", "异常结束时间",
                      "异常类型", "异常指标", "指标数值", "严重等级"]
SEVERITY = {"持续故障": "紧急", "绝对阈值": "重要", "波动": "一般"}
# ====================================================================


def load_data(path: Path) -> pd.DataFrame:
    """读入 CSV，并做字段标准化 + 类型转换（脏数据由 clean_data 处理）。"""
    df = pd.read_csv(path, encoding=ENCODING)
    raw_rows = len(df)
    df = df.rename(columns=RENAME_MAP)                       # 1. 列名标准化
    df["小区ID"] = df["小区ID"].astype(str).str.strip()        # 2. 去掉 ID 首尾空格
    df["时间"] = pd.to_datetime(df["时间"], errors="coerce")   # 3. 解析时间，无法解析的置 NaT
    for col in VALUE_COLUMNS:                                # 4. 指标统一转数值
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"读入 {raw_rows} 行 x {len(df.columns)} 列：{path.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """标准化清洗：删空值 -> 删重复 -> 删负值 -> 裁剪字段。"""
    log = []

    def step(name, before, after):
        log.append((name, before - after, len(df) - after))
        return after

    if "时间" not in df.columns or df["时间"].isna().all():
        raise SystemExit("时间列解析失败，请检查原始文件的时间格式")

    n0 = len(df)
    # 1. 删除核心字段为空的行（对应采集失败、南向接口超时）
    df = df.dropna(subset=CORE_COLUMNS)
    n1 = len(df)
    # 2. 删除完全重复行（重复上报、多次导出叠加）
    df = df.drop_duplicates()
    n2 = len(df)
    # 3. 删除“小区 + 时间”主键重复行，保留第一条
    df = df.drop_duplicates(subset=["小区ID", "时间"], keep="first")
    n3 = len(df)
    # 4. 过滤指标为负的行（计数器回卷、厂家网管缺陷）
    df = df[(df[VALUE_COLUMNS] >= 0).all(axis=1)]
    n4 = len(df)
    # 5. 只保留核心字段，并统一时间格式、统一保留 2 位小数
    df = df[CORE_COLUMNS].copy()
    df["时间"] = df["时间"].dt.strftime(TIME_FMT)
    df["时间"] = pd.to_datetime(df["时间"], format=TIME_FMT)
    df[VALUE_COLUMNS] = df[VALUE_COLUMNS].round(2)
    df = df.sort_values(["小区ID", "时间"]).reset_index(drop=True)

    print("清洗过程：")
    rows = [("删除空值行", n0 - n1), ("删除完全重复行", n1 - n2),
            ("删除主键重复行", n2 - n3), ("过滤负值行", n3 - n4)]
    for name, cnt in rows:
        print(f"  {name:<10} -{cnt} 行")
    print(f"  合计 -{n0 - n4} 行，{(n0 - n4) / n0:.2%}；剩余 {n4} 行 x {df.shape[1]} 列")
    if n0 == n4:
        print("  提示：输入已是清洗后的数据，剔除数为 0 属正常")
    return df


def build_segments(df: pd.DataFrame, hit: pd.Series, col: str, direction: str) -> pd.DataFrame:
    """把命中行按“同小区 + 严格连续 1 小时”切成片段，取片段内最差值。"""
    sub = df.loc[hit, ["小区ID", "时间", col]].sort_values(["小区ID", "时间"])
    rows = []
    for cid, g in sub.groupby("小区ID", sort=False):
        times = g["时间"].tolist()
        vals = g[col].tolist()
        start = prev = times[0]
        best = vals[0]
        for t, v in zip(times[1:], vals[1:]):
            if t - prev == pd.Timedelta(hours=1):        # 与上一小时严格相邻才算同一片段
                best = min(best, v) if direction == "lt" else max(best, v)
            else:                                        # 出现采集缺口，前一段就此闭合
                rows.append((cid, start, prev, best))
                start, best = t, v
            prev = t
        rows.append((cid, start, prev, best))
    return pd.DataFrame(rows, columns=["小区ID", "异常开始时间", "异常结束时间", "指标数值"])


def detect_absolute(df: pd.DataFrame):
    """R1 + R3：先切段，>=3 小时的判持续故障（紧急），其余判绝对阈值（重要）。"""
    tickets, hit_mask = [], pd.Series(False, index=df.index)
    for col, direction, limit in ABS_RULES:
        hit = df[col] < limit if direction == "lt" else df[col] > limit
        hit_mask |= hit
        seg = build_segments(df, hit, col, direction)
        if seg.empty:
            continue
        hours = (seg["异常结束时间"] - seg["异常开始时间"]).dt.total_seconds() / 3600 + 1
        seg["小时数"] = hours.astype(int)
        # 同一故障只出一张单：连续 3 小时以上只出“持续故障”，不再重复出“绝对阈值”
        seg["异常类型"] = np.where(seg["小时数"] >= PERSIST_HOURS, "持续故障", "绝对阈值")
        seg["异常指标"] = col
        tickets.append(seg)
    abs_df = pd.concat(tickets, ignore_index=True) if tickets else pd.DataFrame()
    return abs_df, hit_mask


def detect_wave(df: pd.DataFrame, abs_mask: pd.Series) -> pd.DataFrame:
    """R2：单小时指标劣于该小区当日均值的 70%（掉话率高于 130%），已命中 R1 的小时跳过。"""
    df = df.copy()
    df["日期"] = df["时间"].dt.normalize()          # 归一到自然日，作为“当日”的分组键
    tickets = []
    for col, direction, ratio in [("接入成功率", "lt", WAVE_RATIO),
                                  ("切换成功率", "lt", WAVE_RATIO),
                                  ("掉话率", "gt", WAVE_RATIO_UPPER)]:
        g = df.groupby(["小区ID", "日期"])[col]
        day_mean = g.transform("mean")               # 该小区当日均值
        day_count = g.transform("size")              # 该小区当日有效小时数
        # 样本太少时均值没有参考意义，直接跳过；同时排除已被 R1 命中的小时
        valid = (day_count >= MIN_HOURS_PER_DAY) & (~abs_mask)
        hit = (df[col] < day_mean * ratio) if direction == "lt" else (df[col] > day_mean * ratio)
        hit &= valid
        seg = build_segments(df, hit, col, direction)
        if seg.empty:
            continue
        seg["小时数"] = ((seg["异常结束时间"] - seg["异常开始时间"]).dt.total_seconds() / 3600 + 1).astype(int)
        seg["异常类型"] = "波动"
        seg["异常指标"] = col
        tickets.append(seg)
    return pd.concat(tickets, ignore_index=True) if tickets else pd.DataFrame()


def build_work_order(target: Path, src: Path):
    """主流程：清洗 -> 检测 -> 统计 -> 导出。"""
    df = clean_data(load_data(src))
    print(f"有效数据：{len(df)} 行，小区 {df['小区ID'].nunique()} 个，"
          f"时间 {df['时间'].min():%Y-%m-%d %H:%M} ~ {df['时间'].max():%Y-%m-%d %H:%M}")

    abs_df, abs_mask = detect_absolute(df)
    wave_df = detect_wave(df, abs_mask)
    tickets = pd.concat([abs_df, wave_df], ignore_index=True)
    tickets["严重等级"] = tickets["异常类型"].map(SEVERITY)
    tickets = tickets.sort_values(["异常开始时间", "小区ID", "异常指标"]).reset_index(drop=True)

    # 工单字段严格 7 列，顺序固定
    out = tickets[WORK_ORDER_COLUMNS].copy()
    out["异常开始时间"] = out["异常开始时间"].dt.strftime(TIME_FMT)
    out["异常结束时间"] = out["异常结束时间"].dt.strftime(TIME_FMT)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, encoding=ENCODING)

    print("\n工单总数：{} 张".format(len(out)))
    print(out["严重等级"].value_counts().to_string())

    # 统计1：三类异常占比（数量 + 影响时长）
    print("\n【统计1】三类异常占比")
    stat = tickets.groupby("异常类型").agg(工单数=("小时数", "size"), 累计小时数=("小时数", "sum"))
    stat["数量占比"] = (stat["工单数"] / stat["工单数"].sum()).round(3)
    stat["时长占比"] = (stat["累计小时数"] / stat["累计小时数"].sum()).round(3)
    print(stat.sort_values("累计小时数", ascending=False).to_string())

    # 把每张工单展开成“逐小时”明细，供时段分布和时长统计使用
    hourly = pd.concat([
        pd.DataFrame({"小区ID": r.小区ID,
                      "时间": pd.date_range(r.异常开始时间, r.异常结束时间, freq="h")})
        for r in tickets.itertuples(index=False)], ignore_index=True)
    hourly = hourly.drop_duplicates()          # 同一小时多指标同时异常只算 1 次

    # 统计2：异常时段分布（按小时统计异常小区小时数）
    hour_stat = hourly["时间"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    print("\n【统计2】异常时段分布（Top5 小时）")
    print(hour_stat.sort_values(ascending=False).head(5).to_string())

    # 统计3：异常时长 Top10 小区
    dur = hourly.groupby("小区ID").size().rename("异常小时数")
    cnt = tickets.groupby("小区ID").size().rename("工单数")
    top10 = pd.concat([dur, cnt], axis=1).fillna(0).astype(int).sort_values("异常小时数", ascending=False)
    print("\n【统计3】异常小区 Top10（按异常时长）")
    print(top10.head(10).to_string())
    print(f"\n工单已写入：{target}")


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    build_work_order(OUT_FILE, src)
