"""Release candidate entrypoint. Each search owns its data, artifacts and experience."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import re
from pathlib import Path
import shutil
import uuid

from agentic_pruning.driver import run_loop
from agentic_pruning.e2e.prompt_parser import parse
from agentic_pruning.integration.lib_modules import build_lib_module_set
from autoprune.selection import select_winner

DATA_DIR = Path(__file__).resolve().parents[1] / "agentic_pruning" / "data"


def stage_checkpoint(full_ckpt, work_dir):
    """Give upstream code editable metadata without altering the source checkpoint."""
    staged = Path(work_dir).expanduser().resolve() / f"input_checkpoint_{uuid.uuid4().hex}"
    staged.mkdir(parents=True)
    for src in Path(full_ckpt).resolve().iterdir():
        if src.is_file():
            if src.suffix in {".safetensors", ".bin", ".pt"}:
                (staged / src.name).symlink_to(src.resolve())
            else:
                shutil.copyfile(src, staged / src.name)
    return staged


def make_backend(name, *, full_ckpt, suite, num_trials, work_dir=None, dense_sr=None, plugin_spec=None, backend_options=None, max_eval_tasks=None, prompt=""):
    if plugin_spec:
        from autoprune.backends.loader import load_plugin
        from autoprune.backends.generic_backend import GenericVLABackend
        plugin = load_plugin(plugin_spec, backend_options)
        if suite is None:
            matches = [s for s in plugin.suites if re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',prompt,re.I)]
            if len(matches) > 1:
                raise ValueError("One search supports one suite; select one with --task-suite")
            hint = parse(prompt)
            suite = matches[0] if matches else (hint.suite if hint.suite_specified else plugin.default_suite)
        if plugin.name != name:
            raise ValueError(f"--backend must match the plugin's declared name: {plugin.name}")
        if not plugin.synthetic:
            if not full_ckpt:
                raise ValueError("A real plugin requires --full-ckpt")
            if dense_sr is None or not 0 <= dense_sr <= 1:
                raise ValueError("A real plugin requires a measured --dense-sr")
        backend = GenericVLABackend(plugin, full_ckpt=full_ckpt or "synthetic", suite=suite,
            num_trials=num_trials, work_dir=work_dir, dense_sr=dense_sr, max_eval_tasks=max_eval_tasks)
        if hasattr(plugin, "preflight"):
            plugin.preflight(full_ckpt)
        backend.synthetic = bool(plugin.synthetic)
        return backend
    if name == "mock":
        from recovery_lib.backends import MockRecoveryBackend
        backend = MockRecoveryBackend(n_layers=32, seed=0)
        backend.work_dir = Path(work_dir or "runs/mock")
        backend.synthetic = True
        return backend
    if name != "oft":
        raise ValueError("Built-ins are mock and oft; use --plugin module:PluginClass for another model")
    if not full_ckpt:
        raise ValueError("OFT requires --full-ckpt pointing to a local checkpoint")
    if dense_sr is None or not 0 <= dense_sr <= 1:
        raise ValueError("Pass --dense-sr with the measured baseline for this checkpoint and evaluation protocol")
    from autoprune.backends.oft_plugin import OftPlugin
    from autoprune.backends.generic_backend import GenericVLABackend
    try:
        plugin = OftPlugin(**(backend_options or {}))
    except TypeError as exc:
        raise ValueError(f"Unknown OFT backend configuration: {exc}") from exc
    plugin.active_suite = suite
    plugin.preflight(full_ckpt)
    # Upstream loaders update config/modeling files. Give them run-local metadata;
    # source weight tensors are referenced read-only and never exported as links.
    staged = stage_checkpoint(full_ckpt, work_dir)
    return GenericVLABackend(plugin, full_ckpt=str(staged), suite=suite, num_trials=num_trials,
                             work_dir=work_dir, dense_sr=dense_sr, max_eval_tasks=max_eval_tasks)


def make_agent(name,options=None):
    options = options or {}
    allowed={"model","base_url","max_tokens","timeout","temperature"}
    if set(options)-allowed:raise ValueError("Agent config accepts model/endpoint/budgets only; use environment variables for credentials")
    if name == "rule":
        if options:raise ValueError("The rule demonstration has no agent options")
        from agentic_pruning.agents.rule_agent import RuleAgent
        return RuleAgent()
    if name in {"gpt","openai"}:
        from agentic_pruning.agents.gpt_agent import GptAgent
        cls=GptAgent
    elif name=="claude":
        from agentic_pruning.agents.claude_agent import ClaudeAgent
        cls=ClaudeAgent
    elif name=="qwen":
        from agentic_pruning.agents.qwen_agent import QwenAgent
        cls=QwenAgent
    else:raise ValueError(f"Unknown agent {name}")
    try:return cls(**options)
    except TypeError as exc:raise ValueError(f"Unsupported {name} agent configuration: {exc}") from exc


def run_prompt(prompt, *, backend_name="mock", agent_name="rule", n_iters=None,
               full_ckpt=None, task_suite=None, num_trials=10, recovery_steps=120,
               sr_threshold=None, min_compression=None, data_dir=DATA_DIR, run_root=None,
               dense_sr=None, plugin_spec=None, backend_options=None, max_eval_tasks=None, agent_options=None, recoveries=None,
               export_to=None, verify_export=False):
    if export_to is not None:
        export_to = Path(export_to).expanduser().resolve()
        if export_to.exists():
            raise FileExistsError(f"Export destination must be new: {export_to}")
    if verify_export and (export_to is None or backend_name != "oft" or plugin_spec):
        raise ValueError("--verify-export requires --export-to and the built-in OFT backend; custom models use their native numerical verifier")
    if full_ckpt:
        full_ckpt = str(Path(full_ckpt).expanduser().resolve())
    spec = parse(prompt)
    if task_suite:
        spec.suite = task_suite
    if n_iters is not None:
        spec.max_iters = n_iters
    if sr_threshold is not None:
        spec.sr_threshold = sr_threshold
    if min_compression is not None:
        spec.min_compression = min_compression
    if spec.all_suites:
        raise ValueError("One search supports one suite; run a separate search for each requested suite")
    if spec.max_train_hours is not None:
        raise ValueError("Wall-clock training budgets are not enforced; set train_max_steps in --backend-config")
    if not 0 <= spec.min_compression <= 1:
        raise ValueError("Compression must be between 0 and 1")
    if spec.max_iters < 1 or num_trials < 1 or not 0 <= spec.sr_threshold <= 1:
        raise ValueError("Invalid search budget, rollout count or success threshold")
    if backend_name != "mock" and agent_name == "rule" and not plugin_spec:
        raise ValueError("The scripted rule agent is a synthetic demo; choose an LLM agent for a real search")
    run_dir = Path(run_root or "runs").expanduser().resolve() / f"search_{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True)
    runtime_data = run_dir / "data"
    runtime_data.mkdir()
    for src in Path(data_dir).glob("*.json"):
        shutil.copyfile(src, runtime_data / src.name)
    config_path = runtime_data / "mask_search_config.json"
    cfg = json.loads(config_path.read_text())
    cfg.update(tried_masks=[], archive={})
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    requested_suite = None if plugin_spec and not task_suite else spec.suite
    backend = make_backend(backend_name, full_ckpt=full_ckpt, suite=requested_suite,
                           num_trials=num_trials, work_dir=run_dir / "execution", dense_sr=dense_sr,
                           plugin_spec=plugin_spec, backend_options=backend_options, max_eval_tasks=max_eval_tasks, prompt=prompt)
    if plugin_spec:
        spec.suite = backend.suite
    synthetic = bool(getattr(backend, "synthetic", False))
    if not synthetic and agent_name == "rule":
        raise ValueError("The scripted rule agent is a synthetic demo; choose an LLM agent for a real search")
    if export_to is not None and backend_name != "oft":
        from autoprune.backends.plugin import ModelPlugin
        plugin = getattr(backend, "plugin", None)
        if plugin is None or type(plugin).export_checkpoint is ModelPlugin.export_checkpoint:
            raise ValueError("Automatic delivery requires a model plugin with a native export_checkpoint implementation")
    modules = build_lib_module_set(runtime_data, backend, recovery_steps=recovery_steps)
    if recoveries is not None:
        modules.recovery.restrict(recoveries)
    agent = make_agent(agent_name,agent_options)
    mode = getattr(getattr(backend, "plugin", None), "pruning_mode", "contiguous")
    request = {"backend": backend_name, "plugin": plugin_spec, "synthetic": synthetic, "agent": agent_name,
               "checkpoint": str(Path(full_ckpt).resolve()) if full_ckpt else None,
               "allowed_recoveries": [r["id"] for r in modules.recovery.list_recipes()], "spec": asdict(spec), "rollouts_per_task": num_trials, "pruning_mode": mode,
               "probe_metric": "paper_l2_v1", "dedup_key": "mask+recovery",
               "experience": "fresh per run", "disk_probe_cache_reuse": False,
               "historical_replay": False,
               "delivery_requested":{"destination":str(export_to) if export_to is not None else None,
                                     "native_action_check":bool(verify_export)},
               "agent_config":{"model":getattr(agent,"model",None),"provider":agent_name,"max_tokens":getattr(agent,"max_tokens",None),"timeout":getattr(agent,"timeout",None)},
               "plugin_options":backend_options or {},
               "backend_config":getattr(getattr(backend,"plugin",None),"runtime_config",lambda:backend_options or {})(),
               "evaluation_scope":asdict(backend.scope) if hasattr(backend,"scope") else {"num_trials":num_trials}}
    (run_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    print(f"[search] suite={spec.suite}; SR>={spec.sr_threshold:g}; minimum compression={spec.min_compression:g}; "
          f"min_k={spec.min_k}; max_k={spec.max_k}; iterations={spec.max_iters}",flush=True)
    objective = (f"Remove as many layers as possible with SR >= {spec.sr_threshold:.4f} on {spec.suite}. "
                 f"Use pruning mode {mode}; depth={backend.n_layers}. Test recovery alternatives for a "
                 "fixed structure when appropriate. A prior outcome is not a new evaluation.")
    summary = run_loop(agent, modules, spec.max_iters, run_dir, extra_seed_rules=["[USER REQUEST] " + prompt, objective],
                       pruning_mode=mode, dedup_key="mask+recovery", experience_dir=run_dir / "experience")
    summary["synthetic"] = synthetic
    summary["agent_calls"] = getattr(agent,"call_log",[])
    (run_dir / "trace.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    selection = select_winner(summary, spec, backend)
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    out = {"prompt": prompt, "spec": spec, "summary": summary, "selection": selection,
           "run_dir": str(run_dir)}
    from autoprune.report import write_report
    out["report"] = write_report(out, model=backend_name)
    if export_to is not None:
        delivery = {"status":"no_qualifying_checkpoint", "synthetic":synthetic,
                    "checkpoint":None, "action_equivalence_verified":False,
                    "full_robot_performance_verified":False}
        try:
            if selection["winner_meets_spec"]:
                from autoprune.finalize import finalize
                from autoprune.check_export import verify_files
                finalize(run_dir, export_to)
                manifest = verify_files(export_to)
                delivery.update(status="exported", checkpoint=str(export_to),
                                files_verified=len(manifest["files"]))
                if verify_export:
                    from autoprune.verify_oft_export import verify
                    checked, result = verify(export_to, suite=spec.suite, samples=2,
                                             run_root=run_dir / "export_verification")
                    delivery.update(status="exported_and_action_verified",action_equivalence_verified=True,
                                    verification_dir=str(checked),max_abs_action_error=result["max_abs_action_error"])
        except Exception as exc:
            delivery.update(status="failed",error=f"{type(exc).__name__}: {str(exc)[:500]}")
            (run_dir / "delivery.json").write_text(json.dumps(delivery,indent=2)+"\n")
            raise RuntimeError(f"Search records were saved, but delivery failed; inspect {run_dir / 'delivery.json'}") from exc
        (run_dir / "delivery.json").write_text(json.dumps(delivery,indent=2)+"\n")
        out["delivery"] = delivery
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--backend", default="mock", help="mock, oft, or the name of an explicit custom plugin")
    parser.add_argument("--plugin", help="Developer-owned Python module:PluginClass (imports executable code)")
    parser.add_argument("--agent", choices=["rule", "gpt", "claude", "qwen", "openai"], default="rule")
    parser.add_argument("--full-ckpt")
    parser.add_argument("--task-suite", help="Suite identifier declared by the selected model plugin")
    parser.add_argument("--dense-sr", type=float)
    parser.add_argument("--backend-config",type=Path,help="JSON object passed to the OFT or custom plugin constructor; recorded in request.json")
    parser.add_argument("--agent-config",type=Path,help="JSON provider model/endpoint/token/timeout options; credentials stay in the environment")
    parser.add_argument("--max-eval-tasks",type=int,help="Explicit bounded evaluation for smoke checks")
    parser.add_argument("--recoveries",nargs="+",help="Explicit subset of model recovery recipes, e.g. direct or adapter+head+proprio")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--n-iters", type=int)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--sr-threshold", type=float)
    parser.add_argument("--min-compression", type=float)
    parser.add_argument("--export-to",type=Path,help="Automatically export a qualifying winner to a new directory")
    parser.add_argument("--verify-export",action="store_true",help="After automatic OFT export, compare native actions in independent processes")
    args = parser.parse_args()
    try:
        options=json.loads(args.backend_config.read_text()) if args.backend_config else {}
        agent_options=json.loads(args.agent_config.read_text()) if args.agent_config else {}
        if not isinstance(agent_options,dict):raise ValueError("--agent-config must contain a JSON object")
        if not isinstance(options,dict):raise ValueError("--backend-config must contain a JSON object")
        if args.max_eval_tasks is not None and args.max_eval_tasks<1:raise ValueError("--max-eval-tasks must be positive")
        out = run_prompt(args.prompt, backend_name=args.backend, agent_name=args.agent,
            full_ckpt=args.full_ckpt, task_suite=args.task_suite, dense_sr=args.dense_sr,
            num_trials=args.num_trials, n_iters=args.n_iters, run_root=args.run_root,
            sr_threshold=args.sr_threshold, min_compression=args.min_compression, plugin_spec=args.plugin, backend_options=options, max_eval_tasks=args.max_eval_tasks, agent_options=agent_options, recoveries=args.recoveries,
            export_to=args.export_to,verify_export=args.verify_export)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"AutoPruner: {exc}\n")
    print(json.dumps({"run_dir":out["run_dir"], "synthetic":out["summary"]["synthetic"],
                      "winner_meets_spec":out["selection"]["winner_meets_spec"],
                      **({"delivery":out["delivery"]} if "delivery" in out else {})}, indent=2))
    if "delivery" in out and out["delivery"]["status"] == "no_qualifying_checkpoint":
        return 3
    return 0 if any(row.get("sr") is not None for row in out["summary"]["trace"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
