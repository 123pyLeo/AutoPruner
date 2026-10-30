"""Explicit loading of a developer-provided ModelPlugin, without editing the core."""
from importlib import import_module
import inspect

from .plugin import ModelPlugin


def load_plugin(spec: str, options=None) -> ModelPlugin:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Use --plugin package.module:PluginClass")
    try:
        cls = getattr(import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import the explicit plugin {spec}: {exc}") from exc
    if not inspect.isclass(cls) or not issubclass(cls, ModelPlugin):
        raise ValueError("PluginClass must subclass ModelPlugin")
    if inspect.isabstract(cls):
        raise ValueError("PluginClass must implement all ModelPlugin methods")
    if options is not None and not isinstance(options, dict):
        raise ValueError("Plugin configuration must be a JSON object")
    try:
        plugin = cls(**(options or {}))
    except TypeError as exc:
        raise ValueError(f"Unsupported plugin constructor configuration: {exc}") from exc
    if not isinstance(plugin.name, str) or not plugin.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in plugin.name):
        raise ValueError("Plugin name must contain letters, digits, underscores or hyphens")
    if plugin.name in {"unnamed", "mock", "oft"}:
        raise ValueError("Choose a distinct plugin name; built-in backends cannot be replaced")
    if isinstance(plugin.n_layers, bool) or not isinstance(plugin.n_layers, int) or plugin.n_layers < 2:
        raise ValueError("Declare the actual backbone layer count")
    if plugin.pruning_mode not in {"contiguous", "skip"}:
        raise ValueError("Declare the pruning geometry explicitly")
    if not isinstance(plugin.synthetic, bool):
        raise ValueError("Declare synthetic as a boolean")
    for field in ("suites", "recovery_strategies"):
        values = getattr(plugin, field)
        if (not isinstance(values, (tuple, list)) or not values or
                any(not isinstance(v, str) or not v.strip() for v in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"Declare a nonempty, unique sequence of {field}")
    return plugin
