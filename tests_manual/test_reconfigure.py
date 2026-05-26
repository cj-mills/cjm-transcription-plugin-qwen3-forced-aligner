"""CR-4 reconfigure-lifecycle validation for the Qwen3 Forced Aligner plugin.

Contract-level (no real model load — the model is large/GPU). Exercises the
substrate's reconfigure delta path in-process with a fake model object:

  1. reconfigure(model_id flip) -> RELEASE the model (RELOAD_TRIGGER ->
     _release_model) + RE-APPLY config (_apply_config)
  2. on_disable releases (CR-2)

Requires the substrate version with the two-phase reconfigure (CR-4). Run from the
repo root in the plugin's env:

    conda run -n cjm-transcription-plugin-qwen3-forced-aligner --no-capture-output python tests_manual/test_reconfigure.py
"""
import sys

A = {"model_id": "test/model-a", "device": "cpu", "dtype": "float32"}
B = {"model_id": "test/model-b", "device": "cpu", "dtype": "float32"}


def main() -> int:
    from cjm_transcription_plugin_qwen3_forced_aligner.plugin import Qwen3ForcedAlignerPlugin

    p = Qwen3ForcedAlignerPlugin()
    p._apply_config(A)
    assert p._config.model_id == "test/model-a"

    # 1) model_id trigger: release + re-apply
    p._model = object()
    p.reconfigure(A, B)
    assert p._model is None, "RELOAD_TRIGGER must fire _release_model on model_id change"
    assert p._config.model_id == "test/model-b", "reconfigure must re-apply config (CR-4)"
    print("[1] reconfigure model_id a->b: released + applied  OK")

    # 2) on_disable releases (CR-2)
    p._model = object()
    p.on_disable()
    assert p._model is None, "on_disable must release the model"
    print("[2] on_disable: model released  OK")

    print("RECONFIGURE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
