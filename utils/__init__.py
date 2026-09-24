# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from lumalabs/tvm under the license in
# https://github.com/lumalabs/tvm/blob/main/LICENSE
# Sources:
# https://github.com/lumalabs/tvm/blob/main/dnnlib/util.py
# https://github.com/lumalabs/tvm/blob/main/torch_utils/misc.py
# https://github.com/lumalabs/tvm/blob/main/torch_utils/persistence.py
# Adjusted and consolidated for this project.
import copy
import glob
import hashlib
import html
import http.cookiejar
import importlib
import inspect
import io
import os
import pickle
import re
import sys
import tempfile
import types
import urllib.parse
import urllib.request
import uuid
from typing import Any, Optional

import numpy as np
import torch


class Resolver:
    """Resolve and instantiate Python objects from fully qualified names."""

    def _normalize_path(self, object_path: str) -> str:
        object_path = re.sub(r"^np\.", "numpy.", object_path)
        object_path = re.sub(r"^tf\.", "tensorflow.", object_path)
        return object_path

    def _walk_attributes(self, root: Any, attribute_path: str) -> Any:
        if attribute_path == "":
            return root
        current = root
        for part in attribute_path.split("."):
            current = getattr(current, part)
        return current

    def resolve(self, object_path: str) -> Any:
        normalized_path = self._normalize_path(object_path)
        parts = normalized_path.split(".")
        candidates = [
            (".".join(parts[:index]), ".".join(parts[index:]))
            for index in range(len(parts), 0, -1)
        ]

        for module_name, attribute_path in candidates:
            try:
                module = importlib.import_module(module_name)
                return self._walk_attributes(module, attribute_path)
            except Exception:
                pass

        for module_name, _ in candidates:
            try:
                importlib.import_module(module_name)
            except ImportError:
                exc = sys.exc_info()[1]
                if not str(exc).startswith(f"No module named '{module_name}'"):
                    raise

        raise ImportError(normalized_path)

    def instantiate(self, *args, class_name: Optional[str] = None, **kwargs) -> Any:
        if class_name is None:
            raise ValueError("class_name must be provided")
        target = self.resolve(class_name)
        if not callable(target):
            raise TypeError(f"{class_name} does not resolve to a callable")
        return target(*args, **kwargs)


class ArtifactStore:
    """Manage cache paths and opening local or remote artifacts."""

    def __init__(self) -> None:
        self._cache_root_override: Optional[str] = None

    def cache_path(self, *parts: str) -> str:
        base_dir = self._cache_root_override
        if base_dir is None:
            if "DNNLIB_CACHE_DIR" in os.environ:
                base_dir = os.environ["DNNLIB_CACHE_DIR"]
            elif "HOME" in os.environ:
                base_dir = os.path.join(os.environ["HOME"], ".cache", "dnnlib")
            elif "USERPROFILE" in os.environ:
                base_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "dnnlib")
            else:
                base_dir = os.path.join(tempfile.gettempdir(), ".cache", "dnnlib")
        return os.path.join(base_dir, *parts)

    def _is_remote_url(self, candidate: Any, allow_file_urls: bool = False) -> bool:
        if not isinstance(candidate, str) or "://" not in candidate:
            return False
        if allow_file_urls and candidate.startswith("file://"):
            return True
        try:
            parsed = urllib.parse.urlparse(candidate)
            if not parsed.scheme or not parsed.netloc or "." not in parsed.netloc:
                return False
            root = urllib.parse.urlparse(urllib.parse.urljoin(candidate, "/"))
            return bool(root.scheme and root.netloc and "." in root.netloc)
        except Exception:
            return False

    def open(
        self,
        location: str,
        cache_dir: Optional[str] = None,
        num_attempts: int = 10,
        verbose: bool = True,
        return_filename: bool = False,
        cache: bool = True,
    ) -> Any:
        if num_attempts < 1:
            raise ValueError("num_attempts must be at least 1")
        if return_filename and not cache:
            raise ValueError("return_filename=True requires cache=True")

        if not re.match(r"^[a-z]+://", location):
            return location if return_filename else open(location, "rb")

        if location.startswith("file://"):
            filename = urllib.parse.urlparse(location).path
            if re.match(r"^/[a-zA-Z]:", filename):
                filename = filename[1:]
            return filename if return_filename else open(filename, "rb")

        if not self._is_remote_url(location):
            raise ValueError(f"Unsupported URL: {location}")

        download_dir = cache_dir or self.cache_path("downloads")
        url_digest = hashlib.md5(location.encode("utf-8")).hexdigest()
        if cache:
            matches = glob.glob(os.path.join(download_dir, f"{url_digest}_*"))
            if len(matches) == 1:
                filename = matches[0]
                return filename if return_filename else open(filename, "rb")

        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )
        payload = None
        resolved_name = location

        if verbose:
            print(f"Downloading {location} ...", end="", flush=True)

        for attempts_left in reversed(range(num_attempts)):
            try:
                with opener.open(location) as response:
                    payload = response.read()
                    if not payload:
                        raise IOError("No data received")

                    if len(payload) < 8192:
                        content = payload.decode("utf-8")
                        if "download_warning" in response.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                location = urllib.parse.urljoin(location, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content:
                            raise IOError(
                                "Google Drive download quota exceeded -- please try again later"
                            )

                    match = re.search(
                        r'filename="([^"]*)"',
                        response.headers.get("Content-Disposition", ""),
                    )
                    resolved_name = match[1] if match else location
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except Exception:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

        if cache:
            safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", resolved_name)[:128]
            os.makedirs(download_dir, exist_ok=True)
            cache_file = os.path.join(download_dir, f"{url_digest}_{safe_name}")
            temp_file = os.path.join(
                download_dir,
                f"tmp_{uuid.uuid4().hex}_{url_digest}_{safe_name}",
            )
            with open(temp_file, "wb") as handle:
                handle.write(payload)
            os.replace(temp_file, cache_file)
            return cache_file if return_filename else open(cache_file, "rb")

        return io.BytesIO(payload)


class TensorTools:
    """Tensor-related helpers with internal memoization."""

    def __init__(self) -> None:
        self._constant_cache = {}

    def constant(
        self,
        value: Any,
        shape=None,
        dtype=None,
        device=None,
        memory_format=None,
    ) -> torch.Tensor:
        array = np.asarray(value)
        target_shape = tuple(shape) if shape is not None else None
        target_dtype = dtype or torch.get_default_dtype()
        target_device = device or torch.device("cpu")
        target_memory_format = memory_format or torch.contiguous_format

        key = (
            array.shape,
            array.dtype,
            array.tobytes(),
            target_shape,
            target_dtype,
            target_device,
            target_memory_format,
        )
        tensor = self._constant_cache.get(key)
        if tensor is None:
            tensor = torch.as_tensor(array.copy(), dtype=target_dtype, device=target_device)
            if target_shape is not None:
                tensor, _ = torch.broadcast_tensors(tensor, torch.empty(target_shape))
            tensor = tensor.contiguous(memory_format=target_memory_format)
            self._constant_cache[key] = tensor
        return tensor

    def constant_like(
        self,
        reference: torch.Tensor,
        value: Any,
        shape=None,
        dtype=None,
        device=None,
        memory_format=None,
    ) -> torch.Tensor:
        target_dtype = dtype or reference.dtype
        target_device = device or reference.device
        return self.constant(
            value,
            shape=shape,
            dtype=target_dtype,
            device=target_device,
            memory_format=memory_format,
        )

    def copy_module_state(
        self,
        src_module: torch.nn.Module,
        dst_module: torch.nn.Module,
        require_all: bool = False,
    ) -> None:
        if not isinstance(src_module, torch.nn.Module):
            raise TypeError("src_module must be a torch.nn.Module")
        if not isinstance(dst_module, torch.nn.Module):
            raise TypeError("dst_module must be a torch.nn.Module")

        src_state = dict(list(src_module.named_parameters()) + list(src_module.named_buffers()))
        for name, tensor in list(dst_module.named_parameters()) + list(dst_module.named_buffers()):
            if name not in src_state and require_all:
                raise AssertionError(f"Missing tensor '{name}' in source module")
            if name in src_state:
                tensor.copy_(src_state[name])


class PersistenceManager:
    """Decorator-based persistence for classes whose source should travel with pickles."""

    def __init__(self, version: int = 6) -> None:
        self._version = version
        self._decorated_classes = set()
        self._import_hooks = []
        self._module_to_source = {}
        self._source_to_module = {}

    def serializable_class(self, original_class):
        if not isinstance(original_class, type):
            raise TypeError("serializable_class expects a class")
        if self._is_serializable(original_class):
            return original_class

        if original_class.__module__ not in sys.modules:
            raise RuntimeError(f"Module {original_class.__module__} is not loaded")

        original_module = sys.modules[original_class.__module__]
        module_source = self._capture_module_source(original_module)
        manager = self

        class SerializableWrapper(original_class):
            _serialized_module_source = module_source
            _serialized_class_name = original_class.__name__

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                record_init_args = getattr(self, "_record_init_args", True)
                self._init_args = copy.deepcopy(args) if record_init_args else None
                self._init_kwargs = copy.deepcopy(kwargs) if record_init_args else None
                if original_class.__name__ not in original_module.__dict__:
                    raise AssertionError(
                        f"{original_class.__name__} must remain defined in its original module"
                    )
                manager._ensure_pickle_safe(self.__reduce__())

            @property
            def init_args(self):
                if self._init_args is None:
                    raise AssertionError("init_args were not recorded")
                return copy.deepcopy(self._init_args)

            @property
            def init_kwargs(self):
                if self._init_kwargs is None:
                    raise AssertionError("init_kwargs were not recorded")
                return copy.deepcopy(self._init_kwargs)

            def __reduce__(self):
                pieces = list(super().__reduce__())
                pieces += [None] * max(3 - len(pieces), 0)
                if pieces[0] is not _restore_serialized_object:
                    payload = {
                        "kind": "class",
                        "version": manager._version,
                        "module_source": self._serialized_module_source,
                        "class_name": self._serialized_class_name,
                        "state": pieces[2],
                    }
                    pieces[0] = _restore_serialized_object
                    pieces[1] = (payload,)
                    pieces[2] = None
                return tuple(pieces)

        SerializableWrapper.__name__ = original_class.__name__
        SerializableWrapper.__module__ = original_class.__module__
        self._decorated_classes.add(SerializableWrapper)
        return SerializableWrapper

    def register_import_hook(self, hook) -> None:
        if not callable(hook):
            raise TypeError("hook must be callable")
        self._import_hooks.append(hook)

    def _is_serializable(self, obj: Any) -> bool:
        try:
            if obj in self._decorated_classes:
                return True
        except TypeError:
            pass
        return type(obj) in self._decorated_classes

    def _capture_module_source(self, module: types.ModuleType) -> str:
        source = self._module_to_source.get(module)
        if source is None:
            source = inspect.getsource(module)
            self._module_to_source[module] = source
            self._source_to_module[source] = module
        return source

    def _load_embedded_module(self, source: str) -> types.ModuleType:
        module = self._source_to_module.get(source)
        if module is None:
            module_name = f"_serialized_module_{uuid.uuid4().hex}"
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
            self._module_to_source[module] = source
            self._source_to_module[source] = module
            exec(source, module.__dict__)
        return module

    def _restore(self, payload: dict):
        payload = dict(payload)
        if payload.get("state") is None:
            payload["state"] = {}
        for hook in self._import_hooks:
            payload = hook(payload)
            if payload is None:
                raise AssertionError("import hooks must return a payload")

        if payload["version"] != self._version:
            raise AssertionError("Unsupported serialized object version")

        module = self._load_embedded_module(payload["module_source"])
        if payload["kind"] != "class":
            raise AssertionError(f"Unsupported serialized kind: {payload['kind']}")

        original_class = module.__dict__[payload["class_name"]]
        wrapped_class = self.serializable_class(original_class)
        instance = wrapped_class.__new__(wrapped_class)

        setstate = getattr(instance, "__setstate__", None)
        if callable(setstate):
            setstate(payload["state"])
        else:
            instance.__dict__.update(payload["state"])
        return instance

    def _ensure_pickle_safe(self, obj: Any) -> None:
        def strip_runtime_objects(value: Any):
            if isinstance(value, (list, tuple, set)):
                return [strip_runtime_objects(item) for item in value]
            if isinstance(value, dict):
                return [
                    [strip_runtime_objects(key), strip_runtime_objects(item)]
                    for key, item in value.items()
                ]
            if isinstance(value, (str, int, float, bool, bytes, bytearray)):
                return None
            if f"{type(value).__module__}.{type(value).__name__}" in {
                "numpy.ndarray",
                "torch.Tensor",
                "torch.nn.parameter.Parameter",
            }:
                return None
            if self._is_serializable(value):
                return None
            return value

        with io.BytesIO() as handle:
            pickle.dump(strip_runtime_objects(obj), handle)


def _restore_serialized_object(payload: dict):
    return persistence._restore(payload)


resolver = Resolver()
artifacts = ArtifactStore()
tensors = TensorTools()
persistence = PersistenceManager()

__all__ = [
    "ArtifactStore",
    "PersistenceManager",
    "Resolver",
    "TensorTools",
    "artifacts",
    "persistence",
    "resolver",
    "tensors",
]
