"""Optional setup for the documented one-rank candidate environment only."""


def preload_arrow():
    # In the observed Py3.11/Ray/vLLM wheel set, late Arrow native loading after
    # vLLM crashes during actor deserialization. Load it before task/actor imports.
    # This is NOT a core dependency or a general binary-compatibility guarantee.
    import pyarrow

    if pyarrow.__version__ != "21.0.0":
        raise RuntimeError("candidate Ray worker requires the recorded Arrow version")
