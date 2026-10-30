"""Parse a natural-language user request into a structured ProblemSpec.

Light heuristic parsing (regex + keywords). For paper grade we could route
through an LLM, but explicit parser is more debuggable and reproducible.
"""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProblemSpec:
    suite: str = "libero_object"   # libero_object | libero_10 | libero_spatial | ...
    sr_threshold: float = 0.90     # minimum acceptable success rate
    sr_margin_from_dense: Optional[float] = None  # alt: within X pt of dense
    min_compression: float = 0.0   # 0.0-1.0; e.g. 0.875 = remove 87.5% layers.
    # Default 0.0 = NO hard compression floor: "prune as many as possible" is a
    # maximize objective (the selector already prefers most-pruned at SR>=thr), so a
    # prompt that only sets an SR target must NOT silently require k>=16 and then
    # report "no plan met spec". The lightweight/extreme keywords below still raise it.
    max_k: Optional[int] = None    # alt: explicit "remove at most N layers"
    min_k: Optional[int] = None    # explicit "remove at least N layers"
    max_iters: int = 6
    max_train_hours: Optional[float] = None  # budget cap; None = unlimited
    raw_prompt: str = ""
    suite_specified: bool = False  # did the prompt actually NAME a dimension?
    all_suites: bool = False       # run every dimension of the model


def parse(prompt: str) -> ProblemSpec:
    spec = ProblemSpec(raw_prompt=prompt)
    p = prompt.lower()

    # Dimension / suite detection (+ whether one was explicitly named, + "all")
    if (re.search(r"all\s+(?:four\s+)?(?:dimensions?|suites?)", p) or
            re.search(r"(?:全部|所有|每个|四个)(?:的)?(?:任务集|维度|suite)", p)):
        spec.all_suites = True
        spec.suite_specified = True
    elif "libero_object" in p or "libero-object" in p or "object" in p or "物体" in p:
        spec.suite = "libero_object"; spec.suite_specified = True
    elif "libero_goal" in p or "libero-goal" in p or "goal" in p:
        spec.suite = "libero_goal"; spec.suite_specified = True
    elif "libero_10" in p or "libero-10" in p or "libero 10" in p or "long" in p or "长程" in p:
        spec.suite = "libero_10"; spec.suite_specified = True
    elif "libero_spatial" in p or "spatial" in p or "空间" in p:
        spec.suite = "libero_spatial"; spec.suite_specified = True

    # Do not misread a relative degradation budget as an absolute success rate.
    if re.search(r"成功率\s*(?:允许)?(?:下降|降低|损失)|(?:sr|success\s+rate)\s+(?:may\s+|can\s+)?(?:drop|decrease|degrade)", p):
        raise ValueError('Relative success-rate wording is ambiguous; use an absolute SR>=85% or the supported "within X pt of dense" form')
    # An explicit percent sign always means percent, including 1% and 0.5%.
    m = re.search(r"(?:\bsr\b|\bsuccess[ _-]*rate\b)\s*(?:[>≥]=?|above|at\s+least|no\s+less\s+than)\s*(\d+(?:\.\d+)?)\s*(%)?", p)
    if m:
        v = float(m.group(1))
        spec.sr_threshold = v / 100 if m.group(2) or v > 1 else v
    elif re.search(r"成功率[^,，。;；]*?(\d+(?:\.\d+)?)\s*(%)?", p):
        # decimal-aware: "成功率不低于 0.90" must parse as 0.90, not the bare "0"
        m = re.search(r"成功率[^,，。;；]*?(\d+(?:\.\d+)?)\s*(%)?", p)
        v = float(m.group(1))
        spec.sr_threshold = v / 100 if m.group(2) or v > 1 else v

    # "within X pt of dense"
    m = re.search(r"within\s+(\d+(?:\.\d+)?)\s*(pt|points|%)\s+(?:of\s+)?dense", p)
    if m:
        spec.sr_margin_from_dense = float(m.group(1)) / 100

    # Compression: "remove 87.5%" / "压缩 80%" / "k>=24" / "delete N layers"
    m = re.search(r"(?:remove|compress|prune|删|压缩|剪枝)\s*[><≥]*\s*(\d+(?:\.\d+)?)\s*%", p)
    if m:
        spec.min_compression = float(m.group(1)) / 100
    m = re.search(r"k\s*[>≥]=?\s*(\d+)", p)
    if m:
        spec.min_k = int(m.group(1))
    m = re.search(r"k\s*[<≤]=?\s*(\d+)", p)
    if m:
        spec.max_k = int(m.group(1))
    m = re.search(r"(?:最多|至多)(?:删除|删|剪掉|剪|移除)?\s*(\d+)\s*层|(?:remove|delete|prune)\s+at\s+most\s+(\d+)\s+layers", p)
    if m:
        spec.max_k = int(m.group(1) or m.group(2))
    m = re.search(r"(?:至少|最少)(?:删除|删|剪掉|剪|移除)?\s*(\d+)\s*层|(?:remove|delete|prune)\s+at\s+least\s+(\d+)\s+layers", p)
    if m:
        spec.min_k = int(m.group(1) or m.group(2))
    if "lightweight" in p or "轻量" in p or "light" in p:
        # heuristic: lightweight ≈ remove >= 50%
        if spec.min_compression < 0.5:
            spec.min_compression = 0.5
    if "extreme" in p or "极限" in p or "maximum" in p or "最大" in p:
        spec.min_compression = max(spec.min_compression, 0.75)

    # Iters / budget
    m = re.search(r"max[\s_-]?iter[s]?\s*=\s*(\d+)", p)
    if m:
        spec.max_iters = int(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr|小时)", p)
    if m:
        spec.max_train_hours = float(m.group(1))

    return spec


def spec_to_summary(spec: ProblemSpec, n_layers: int = 32) -> str:
    # n_layers comes from the resolved backend so the count is model-correct (OFT 32,
    # SmolVLA 16, StarVLA 36); the ProblemSpec itself is model-agnostic (a fraction).
    parts = [f"suite={spec.suite}",
             f"sr_threshold={spec.sr_threshold:.2f}",
             f"min_compression={spec.min_compression:.2f} "
             f"(= remove ≥{int(round(spec.min_compression*n_layers))} of {n_layers} layers)"]
    if spec.sr_margin_from_dense:
        parts.append(f"sr_margin_from_dense={spec.sr_margin_from_dense:.3f}")
    if spec.min_k:
        parts.append(f"min_k={spec.min_k}")
    if spec.max_k:
        parts.append(f"max_k={spec.max_k}")
    if spec.max_train_hours:
        parts.append(f"max_train_hours={spec.max_train_hours}")
    return ", ".join(parts)


if __name__ == "__main__":
    # smoke
    for q in [
        "give me a lightweight VLA for libero_object with SR > 90%",
        "extreme compression, k>=24, SR within 1pt of dense",
        "请给我一个轻量化的 libero_object 模型，成功率 95%",
    ]:
        s = parse(q)
        print(f"\n  prompt: {q}")
        print(f"  parsed: {spec_to_summary(s)}")
