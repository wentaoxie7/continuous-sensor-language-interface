import torch


def load_transformers_components():
    import transformers

    if not transformers.utils.is_torch_available():
        torch_version = getattr(torch, "__version__", "unknown")
        raise RuntimeError(
            "Transformers cannot access the PyTorch backend in the current environment. "
            f"Detected torch={torch_version} and transformers={transformers.__version__}. "
            "Install a transformers version compatible with your torch build, or upgrade torch."
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer
