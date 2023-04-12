from typing import Callable, Iterable, Optional
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


@dataclass
class HookInfo:
    handle: RemovableHandle
    level: Optional[int] = None


class HookedModule(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._hooks: list[HookInfo] = []
        self.context_level: int = 0

    @contextmanager
    def hooks(
        self,
        fwd: list[tuple[str, Callable]] = [],
        bwd: list[tuple[str, Callable]] = [],
    ):
        """Context manager for registering hooks.

        fwd/bwd: list of tuples (module_path, hook_fn)
        """
        self.context_level += 1
        try:
            # Add hooks
            for hook_position, hook_fn in fwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_forward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            for hook_position, hook_fn in bwd:
                module = self._get_module_by_path(hook_position)
                handle = module.register_full_backward_hook(hook_fn)
                info = HookInfo(handle=handle, level=self.context_level)
                self._hooks.append(info)

            yield self

        finally:
            # Remove hooks
            for info in self._hooks:
                if info.level == self.context_level:
                    info.handle.remove()
            self._hooks = [h for h in self._hooks if h.level != self.context_level]
            self.context_level -= 1

    def _get_module_by_path(self, path: str) -> nn.Module:
        module = self.model
        for attr in path.split("."):
            module = getattr(module, attr)

        return module

    def get_hookable_module_paths(self) -> list[str]:
        """Get all module paths in the model. Modules that can't be hooked are excluded."""
        module_paths = []
        for name, _ in self.model.named_modules():
            if name == "":
                continue
            if isinstance(self._get_module_by_path(name), nn.ModuleList):
                continue
            module_paths.append(name)

        return module_paths

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def print_nested(obj, depth: int = 0) -> None:
        """Print nested objects in a tree-like structure.
        Experimental, not very robust.
        """
        if depth >= 10:
            raise RecursionError("In too deep bro")

        indent = "    "

        if isinstance(obj, torch.Tensor):
            print(f"{indent * depth}{obj.shape}")
        # Might need an elif to catch lists/tuples with a lot of ints/floats
        elif isinstance(obj, str):
            print(f"{indent * depth}str: {obj}")
        elif isinstance(obj, Iterable):
            print(f"{indent * depth}{type(obj).__name__}(")
            for o in obj:
                if isinstance(o, str) and hasattr(obj, o):
                    print(f"{indent * (depth + 1)}.{o}:")
                    HookedModule.print_nested(getattr(obj, o), depth + 2)
                else:
                    HookedModule.print_nested(o, depth + 1)
            print(indent * depth + ")")
        else:
            print(indent * depth + type(obj).__name__)
