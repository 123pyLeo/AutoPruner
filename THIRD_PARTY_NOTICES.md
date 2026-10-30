# Third party notices

This candidate includes code derived from the AutoPrune research workspace and its StarVLA-derived implementation. The original StarVLA MIT notice is retained in LICENSE.

The OFT source overlay is derived from [OpenVLA-OFT](https://github.com/moojink/openvla-oft), whose MIT license is included in `autoprune/integrations/openvla_oft/LICENSE`. The overlay also contains local experimental extensions; their original-source and release checksums are recorded in its manifest.

The private `autoprune._vendor.oft_probe` namespace contains selected StarVLA wrapper, action head, projector, configuration and image utilities. Namespace changes isolate their imports. `autoprune/oft_observations.py` extracts the probe helpers from the experimental script, with strict malformed-sample errors replacing silent padding/skipping. This is an engineering extraction, not an assertion that a new version reproduces all historical behavior.

Base model and dataset distributions retain their own terms. This code license does not relicense external checkpoints, datasets, transitive dependencies or GPU runtime binaries.

The StarVLA reference components and FastWAM MoT reference patch under `examples/` retain their respective MIT licenses. Their fixed public upstream revisions, local source hashes and extraction changes are recorded in `examples/source_manifest.json`. They are not complete training integrations.
