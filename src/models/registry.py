"""Model registry implementing the Strategy pattern for IranBSE models.

Provides a singleton :class:`ModelRegistry` that allows model classes to be
registered by name (either imperatively or via the :func:`register_model`
decorator) and later instantiated through a single factory method.

Typical usage::

    @register_model("regime_switching")
    class RegimeSwitchingModel(BaseModel): ...

    # Later, in application code:
    registry = ModelRegistry()
    model = registry.create("regime_switching", config)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

from loguru import logger

from src.models.base_model import BaseModel, ModelConfig

T = TypeVar("T", bound=type[BaseModel])


class ModelRegistry:
    """Thread-safe singleton registry mapping names to model classes.

    The registry ensures that only one global catalogue exists across the
    entire process, regardless of how many times the constructor is called.

    Example::

        registry = ModelRegistry()
        registry.register("garch", GARCHModel)
        model = registry.create("garch", ModelConfig(name="garch"))
    """

    _instance: ModelRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> ModelRegistry:
        """Return the singleton instance, creating it on first access."""
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking for thread safety.
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._registry: dict[str, type[BaseModel]] = {}
                    cls._instance = instance
                    logger.debug("ModelRegistry singleton created.")
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str, model_class: type[BaseModel]) -> None:
        """Register a model class under *name*.

        Args:
            name: Unique string key for the model (e.g. ``"garch"``).
            model_class: A concrete subclass of :class:`BaseModel`.

        Raises:
            TypeError: If *model_class* is not a subclass of
                :class:`BaseModel`.
            ValueError: If *name* is already registered.
        """
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            raise TypeError(
                f"model_class must be a subclass of BaseModel, "
                f"got {model_class!r}"
            )

        if name in self._registry:
            raise ValueError(
                f"A model is already registered under the name '{name}'. "
                f"Existing: {self._registry[name].__name__}"
            )

        self._registry[name] = model_class
        logger.info("Registered model '{}' -> {}", name, model_class.__name__)

    def create(self, name: str, config: ModelConfig) -> BaseModel:
        """Instantiate a registered model.

        Args:
            name: The key previously passed to :meth:`register`.
            config: Configuration forwarded to the model constructor.

        Returns:
            A new instance of the requested model class.

        Raises:
            KeyError: If *name* is not in the registry.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none)"
            raise KeyError(
                f"No model registered under '{name}'. "
                f"Available models: {available}"
            )

        model_class = self._registry[name]
        logger.debug("Creating model '{}' from class {}", name, model_class.__name__)
        return model_class(config)

    def list_models(self) -> list[str]:
        """Return the sorted list of registered model names.

        Returns:
            Alphabetically sorted list of registered keys.
        """
        return sorted(self._registry)

    def unregister(self, name: str) -> None:
        """Remove a model from the registry.

        Args:
            name: Key to remove.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"No model registered under '{name}'.")
        del self._registry[name]
        logger.info("Unregistered model '{}'", name)

    def clear(self) -> None:
        """Remove all registered models.  Primarily useful in tests."""
        self._registry.clear()
        logger.debug("ModelRegistry cleared.")


# ------------------------------------------------------------------
# Decorator shorthand
# ------------------------------------------------------------------

def register_model(name: str) -> Callable[[T], T]:
    """Class decorator that registers a model with the global registry.

    Args:
        name: The key under which the decorated class will be registered.

    Returns:
        The original class, unmodified.

    Example::

        @register_model("extreme_value")
        class ExtremeValueModel(BaseModel): ...
    """

    def decorator(cls: T) -> T:
        ModelRegistry().register(name, cls)
        return cls

    return decorator
