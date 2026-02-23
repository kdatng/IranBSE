"""Tests for the ModelRegistry singleton and registration decorator.

Verifies the strategy-pattern model registry:
    - Singleton guarantees
    - Registration and creation lifecycle
    - Duplicate/invalid registration errors
    - Decorator-based registration
    - Thread safety
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest

from src.models.base_model import BaseModel, ModelConfig, PredictionResult
from src.models.registry import ModelRegistry, register_model


# ---------------------------------------------------------------------------
# Concrete test models
# ---------------------------------------------------------------------------

class DummyModelA(BaseModel):
    """Minimal model for registry testing."""

    def fit(self, data: pl.DataFrame) -> None:
        """No-op fit.

        Args:
            data: Input data (unused).
        """
        self._mark_fitted(data)

    def predict(self, horizon: int, n_scenarios: int = 1000) -> PredictionResult:
        """Return zeros.

        Args:
            horizon: Forward periods.
            n_scenarios: Simulation count.

        Returns:
            PredictionResult with zero forecasts.
        """
        self._require_fitted()
        zeros = [0.0] * horizon
        return PredictionResult(
            point_forecast=zeros,
            lower_bounds={0.05: zeros},
            upper_bounds={0.95: zeros},
        )

    def get_params(self) -> dict[str, Any]:
        """Return empty params.

        Returns:
            Empty dictionary.
        """
        return {}


class DummyModelB(BaseModel):
    """Second minimal model for registry testing."""

    def fit(self, data: pl.DataFrame) -> None:
        """No-op fit.

        Args:
            data: Input data (unused).
        """
        self._mark_fitted(data)

    def predict(self, horizon: int, n_scenarios: int = 1000) -> PredictionResult:
        """Return ones.

        Args:
            horizon: Forward periods.
            n_scenarios: Simulation count.

        Returns:
            PredictionResult with unit forecasts.
        """
        self._require_fitted()
        ones = [1.0] * horizon
        return PredictionResult(
            point_forecast=ones,
            lower_bounds={0.05: ones},
            upper_bounds={0.95: ones},
        )

    def get_params(self) -> dict[str, Any]:
        """Return model identifier.

        Returns:
            Dictionary with model name.
        """
        return {"model": "B"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_registry() -> None:
    """Clear the registry before and after each test to prevent leakage."""
    registry = ModelRegistry()
    registry.clear()
    yield  # type: ignore[misc]
    registry.clear()


@pytest.fixture
def registry() -> ModelRegistry:
    """Return the singleton ModelRegistry instance."""
    return ModelRegistry()


@pytest.fixture
def sample_config() -> ModelConfig:
    """Create a simple ModelConfig for testing."""
    return ModelConfig(name="test_model", version="1.0.0", params={"lr": 0.01})


# ---------------------------------------------------------------------------
# Tests: Singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    """Tests that ModelRegistry is a proper singleton."""

    def test_same_instance(self) -> None:
        """Multiple instantiations return the same object."""
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2

    def test_shared_state(self) -> None:
        """Registration in one reference is visible in another."""
        r1 = ModelRegistry()
        r2 = ModelRegistry()

        r1.register("shared_model", DummyModelA)
        assert "shared_model" in r2.list_models()


# ---------------------------------------------------------------------------
# Tests: register / create / list_models
# ---------------------------------------------------------------------------

class TestRegistration:
    """Tests for the register, create, and list_models API."""

    def test_register_and_list(self, registry: ModelRegistry) -> None:
        """Registered model appears in list_models."""
        registry.register("model_a", DummyModelA)
        assert "model_a" in registry.list_models()

    def test_register_multiple(self, registry: ModelRegistry) -> None:
        """Multiple models can be registered under different names."""
        registry.register("model_a", DummyModelA)
        registry.register("model_b", DummyModelB)

        models = registry.list_models()
        assert "model_a" in models
        assert "model_b" in models
        assert len(models) == 2

    def test_list_models_sorted(self, registry: ModelRegistry) -> None:
        """list_models returns names in alphabetical order."""
        registry.register("zeta", DummyModelA)
        registry.register("alpha", DummyModelB)
        registry.register("mu", DummyModelA)

        assert registry.list_models() == ["alpha", "mu", "zeta"]

    def test_create_returns_instance(
        self, registry: ModelRegistry, sample_config: ModelConfig
    ) -> None:
        """create() returns an instance of the registered class."""
        registry.register("model_a", DummyModelA)
        model = registry.create("model_a", sample_config)

        assert isinstance(model, DummyModelA)
        assert isinstance(model, BaseModel)

    def test_create_passes_config(
        self, registry: ModelRegistry
    ) -> None:
        """Created model receives the provided config."""
        registry.register("model_a", DummyModelA)
        config = ModelConfig(name="custom", params={"x": 42})
        model = registry.create("model_a", config)

        assert model.config.name == "custom"
        assert model.config.params == {"x": 42}

    def test_create_unknown_model_raises(
        self, registry: ModelRegistry, sample_config: ModelConfig
    ) -> None:
        """Creating an unregistered model raises KeyError."""
        with pytest.raises(KeyError, match="No model registered"):
            registry.create("nonexistent", sample_config)

    def test_create_error_message_lists_available(
        self, registry: ModelRegistry, sample_config: ModelConfig
    ) -> None:
        """KeyError message includes list of available models."""
        registry.register("model_a", DummyModelA)
        try:
            registry.create("bad_name", sample_config)
            pytest.fail("Expected KeyError")
        except KeyError as exc:
            assert "model_a" in str(exc)

    def test_duplicate_name_raises(self, registry: ModelRegistry) -> None:
        """Registering the same name twice raises ValueError."""
        registry.register("dup", DummyModelA)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("dup", DummyModelB)

    def test_non_basemodel_raises(self, registry: ModelRegistry) -> None:
        """Registering a class that is not a BaseModel subclass raises TypeError."""
        with pytest.raises(TypeError, match="subclass of BaseModel"):
            registry.register("bad", str)  # type: ignore[arg-type]

    def test_register_non_class_raises(self, registry: ModelRegistry) -> None:
        """Registering a non-class object raises TypeError."""
        with pytest.raises(TypeError, match="subclass of BaseModel"):
            registry.register("bad", "not_a_class")  # type: ignore[arg-type]

    def test_unregister(self, registry: ModelRegistry) -> None:
        """Unregistering removes the model from the list."""
        registry.register("temp", DummyModelA)
        assert "temp" in registry.list_models()

        registry.unregister("temp")
        assert "temp" not in registry.list_models()

    def test_unregister_unknown_raises(self, registry: ModelRegistry) -> None:
        """Unregistering a non-existent model raises KeyError."""
        with pytest.raises(KeyError):
            registry.unregister("never_registered")

    def test_clear_empties_registry(self, registry: ModelRegistry) -> None:
        """clear() removes all registered models."""
        registry.register("a", DummyModelA)
        registry.register("b", DummyModelB)
        assert len(registry.list_models()) == 2

        registry.clear()
        assert registry.list_models() == []

    def test_empty_registry_lists_nothing(
        self, registry: ModelRegistry
    ) -> None:
        """Empty registry returns empty list."""
        assert registry.list_models() == []


# ---------------------------------------------------------------------------
# Tests: Decorator registration
# ---------------------------------------------------------------------------

class TestDecoratorRegistration:
    """Tests for the @register_model decorator."""

    def test_decorator_registers_class(self) -> None:
        """@register_model registers the class in the global registry."""

        @register_model("decorated_model")
        class DecoratedModel(BaseModel):
            def fit(self, data: pl.DataFrame) -> None:
                self._mark_fitted(data)

            def predict(
                self, horizon: int, n_scenarios: int = 1000
            ) -> PredictionResult:
                return PredictionResult(
                    point_forecast=[0.0] * horizon,
                    lower_bounds={},
                    upper_bounds={},
                )

            def get_params(self) -> dict[str, Any]:
                return {}

        registry = ModelRegistry()
        assert "decorated_model" in registry.list_models()

    def test_decorator_returns_original_class(self) -> None:
        """Decorator does not modify the original class."""

        @register_model("preserved_model")
        class PreservedModel(BaseModel):
            custom_attr = "preserved"

            def fit(self, data: pl.DataFrame) -> None:
                pass

            def predict(
                self, horizon: int, n_scenarios: int = 1000
            ) -> PredictionResult:
                return PredictionResult(
                    point_forecast=[0.0] * horizon,
                    lower_bounds={},
                    upper_bounds={},
                )

            def get_params(self) -> dict[str, Any]:
                return {}

        assert PreservedModel.custom_attr == "preserved"
        assert PreservedModel.__name__ == "PreservedModel"

    def test_decorator_duplicate_raises(self) -> None:
        """Decorator with a duplicate name raises ValueError."""

        @register_model("unique_name")
        class FirstModel(BaseModel):
            def fit(self, data: pl.DataFrame) -> None:
                pass

            def predict(
                self, horizon: int, n_scenarios: int = 1000
            ) -> PredictionResult:
                return PredictionResult(
                    point_forecast=[], lower_bounds={}, upper_bounds={}
                )

            def get_params(self) -> dict[str, Any]:
                return {}

        with pytest.raises(ValueError, match="already registered"):

            @register_model("unique_name")
            class SecondModel(BaseModel):
                def fit(self, data: pl.DataFrame) -> None:
                    pass

                def predict(
                    self, horizon: int, n_scenarios: int = 1000
                ) -> PredictionResult:
                    return PredictionResult(
                        point_forecast=[], lower_bounds={}, upper_bounds={}
                    )

                def get_params(self) -> dict[str, Any]:
                    return {}


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Tests for concurrent access to the registry."""

    def test_concurrent_registration(self, registry: ModelRegistry) -> None:
        """Multiple threads can register different models safely."""
        errors: list[Exception] = []

        def register_model_thread(name: str, model_class: type) -> None:
            try:
                registry.register(name, model_class)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(
                target=register_model_thread,
                args=(f"thread_model_{i}", DummyModelA if i % 2 == 0 else DummyModelB),
            )
            for i in range(10)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.list_models()) == 10

    def test_concurrent_create(self, registry: ModelRegistry) -> None:
        """Multiple threads can create model instances concurrently."""
        registry.register("concurrent_model", DummyModelA)
        results: list[BaseModel] = []
        lock = threading.Lock()

        def create_model() -> None:
            config = ModelConfig(name="test")
            model = registry.create("concurrent_model", config)
            with lock:
                results.append(model)

        threads = [threading.Thread(target=create_model) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert all(isinstance(m, DummyModelA) for m in results)
