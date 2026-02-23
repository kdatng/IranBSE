"""Abstract base class for all feature processors in the IranBSE pipeline.

Processors transform raw fetched data into model-ready features.  Each
processor operates on a Polars DataFrame and returns an enriched DataFrame
with new feature columns appended.  The abstract interface ensures all
processors expose a consistent API for pipeline orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field


class ProcessorConfig(BaseModel):
    """Base configuration for all feature processors.

    Attributes:
        name: Human-readable processor name.
        prefix: Column-name prefix for all features produced by this processor.
        drop_intermediate: Whether to drop intermediate computation columns
            from the output (keeps only final features).
    """

    name: str = Field(..., min_length=1, description="Processor identifier")
    prefix: str = Field(default="", description="Column prefix for generated features")
    drop_intermediate: bool = Field(default=True)

    model_config = {"frozen": False}


class BaseProcessor(ABC):
    """Abstract base class for feature engineering processors.

    Subclasses must implement ``process`` and ``get_feature_names``.  The
    base class provides logging, validation helpers, and a standardised
    ``run`` method that wraps ``process`` with pre/post checks.

    Args:
        config: Pydantic configuration model for the processor.

    Example:
        >>> class MyProcessor(BaseProcessor):
        ...     def process(self, df):
        ...         return df.with_columns(pl.lit(1).alias("my_feature"))
        ...     def get_feature_names(self):
        ...         return ["my_feature"]
    """

    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        logger.info("Initialised processor: {name}", name=config.name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform a DataFrame by adding engineered feature columns.

        Implementations must be **pure** — they should not mutate the input
        DataFrame.  All new columns should be added via ``with_columns`` or
        equivalent non-mutating operations.

        Args:
            df: Input DataFrame (typically from a fetcher or prior processor).

        Returns:
            A new DataFrame containing all original columns plus engineered
            features.  Feature column names should use the configured
            ``prefix`` when set.
        """
        ...

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Return the list of feature column names produced by this processor.

        This is used for downstream pipeline introspection, feature selection,
        and documentation.

        Returns:
            List of column name strings.
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute the processor with pre/post validation and logging.

        This is the recommended entry point for pipeline orchestration.
        It logs input/output shapes, validates that expected features are
        present after processing, and warns about any unexpected nulls.

        Args:
            df: Input DataFrame.

        Returns:
            Processed DataFrame with features appended.

        Raises:
            ValueError: If the processor produces no new columns.
        """
        input_cols = set(df.columns)
        input_rows = len(df)
        logger.debug(
            "[{name}] Processing {rows} rows x {cols} cols",
            name=self.config.name,
            rows=input_rows,
            cols=len(input_cols),
        )

        result = self.process(df)

        new_cols = set(result.columns) - input_cols
        if not new_cols:
            logger.warning(
                "[{name}] Processor added no new columns", name=self.config.name
            )

        # Check for fully-null feature columns.
        for col in new_cols:
            null_count = result.get_column(col).null_count()
            if null_count == len(result):
                logger.warning(
                    "[{name}] Feature '{col}' is entirely null",
                    name=self.config.name,
                    col=col,
                )
            elif null_count > 0:
                null_pct = null_count / len(result) * 100
                logger.debug(
                    "[{name}] Feature '{col}' has {pct:.1f}% nulls",
                    name=self.config.name,
                    col=col,
                    pct=null_pct,
                )

        logger.info(
            "[{name}] Produced {n} features: {features}",
            name=self.config.name,
            n=len(new_cols),
            features=sorted(new_cols),
        )
        return result

    def _prefixed(self, name: str) -> str:
        """Apply the configured prefix to a feature name.

        Args:
            name: Raw feature name.

        Returns:
            Prefixed name if a prefix is configured, otherwise the raw name.
        """
        if self.config.prefix:
            return f"{self.config.prefix}_{name}"
        return name

    @staticmethod
    def _require_columns(df: pl.DataFrame, columns: list[str]) -> None:
        """Assert that all required columns exist in the DataFrame.

        Args:
            df: DataFrame to check.
            columns: List of required column names.

        Raises:
            KeyError: If any column is missing.
        """
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing required columns: {missing}. "
                f"Available: {df.columns}"
            )

    @staticmethod
    def _safe_divide(
        numerator: pl.Expr, denominator: pl.Expr, default: float = 0.0
    ) -> pl.Expr:
        """Safely divide two expressions, returning a default on zero denominator.

        Args:
            numerator: Numerator expression.
            denominator: Denominator expression.
            default: Value to return when the denominator is zero or null.

        Returns:
            Polars expression for the safe division.
        """
        return (
            pl.when(denominator != 0.0)
            .then(numerator / denominator)
            .otherwise(default)
        )

    def get_info(self) -> dict[str, Any]:
        """Return processor metadata for pipeline introspection.

        Returns:
            Dictionary with processor name, class, and feature list.
        """
        return {
            "name": self.config.name,
            "class": type(self).__name__,
            "prefix": self.config.prefix,
            "features": self.get_feature_names(),
            "n_features": len(self.get_feature_names()),
        }
