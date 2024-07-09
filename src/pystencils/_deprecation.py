def _deprecated(feature, instead, version="2.1"):
    from warnings import warn

    warn(
        f"{feature} is deprecated and will be removed in pystencils {version}."
        f"Use {instead} instead.",
        DeprecationWarning,
    )
