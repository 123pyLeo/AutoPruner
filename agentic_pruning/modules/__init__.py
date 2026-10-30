from .probe import ProbeModule
from .probe_query import ProbeQueryService
from .mask_search import MaskSearchModule
from .recovery import RecoveryModule
from .eval import EvalModule
from .diagnosis import DiagnosisModule


class ModuleSet:
    """Bundle of modules + suite metadata. Driver holds one ModuleSet."""
    def __init__(self, probe: ProbeModule, mask_search: MaskSearchModule,
                  recovery: RecoveryModule, eval_: EvalModule, diagnosis: DiagnosisModule,
                  probe_query: ProbeQueryService = None):
        self.probe = probe
        self.mask_search = mask_search
        self.recovery = recovery
        self.eval = eval_
        self.diagnosis = diagnosis
        self.probe_query = probe_query

    @classmethod
    def from_data_dir(cls, data_dir, probe_cache_path=None, recovery_path=None):
        """`recovery_path`: optional override for the recovery-recipe menu file
        (default: <data_dir>/recovery_recipes.json). Added for the open_menu_replay
        experiment (2026-07-16) so an alternate menu (recovery_recipes_open.json)
        can be selected WITHOUT touching the pinned original."""
        from pathlib import Path
        d = Path(data_dir)
        probe = ProbeModule(d / "probe_object.json")
        # Default probe cache is the libero_object pre-computed file.
        if probe_cache_path is None:
            probe_cache_path = d.parent / "experiments/cache/probes_libero_object.json"
        probe_query = ProbeQueryService(
            mode="cached",
            cache_path=Path(probe_cache_path),
            model_id="openvla-7b-oft-finetuned-libero-object",
            suite="libero_object",
        )
        return cls(
            probe=probe,
            mask_search=MaskSearchModule(d / "mask_search_config.json"),
            recovery=RecoveryModule(Path(recovery_path) if recovery_path is not None
                                     else d / "recovery_recipes.json"),
            eval_=EvalModule(d / "eval_lookup_object.json"),
            diagnosis=DiagnosisModule(
                d / "diagnosis_taxonomy.json",
                dense_sr=probe.dense_sr,
                layer_static_facts=probe.get_layer_static_facts(),
                n_layers=probe.n_layers,
            ),
            probe_query=probe_query,
        )
