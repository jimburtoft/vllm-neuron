# SPDX-License-Identifier: Apache-2.0
"""Threshold validator for performance testing."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ThresholdValidator:
    """Validates performance metrics against absolute and regression thresholds."""

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        """Initialize validator with threshold configuration.
        
        Args:
            thresholds: Dictionary containing threshold configuration with keys:
                - absolute: Dict of metric_name -> max_value
                - regression: Dict of metric_name -> max_regression_percent
                - baseline_file: Path to baseline metrics file
        """
        self.thresholds = thresholds or {}

    def validate(self, metrics: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate metrics against configured thresholds.
        
        Args:
            metrics: Current performance metrics
            
        Returns:
            Tuple of (passed, failures, details)
        """
        passed = True
        failures = []
        details = {"absolute": {}, "regression": {}}

        # Validate absolute thresholds
        if "absolute" in self.thresholds:
            abs_passed, abs_failures, abs_details = self._validate_absolute(
                metrics, self.thresholds["absolute"]
            )
            passed = passed and abs_passed
            failures.extend(abs_failures)
            details["absolute"] = abs_details

        # Validate regression thresholds
        if "regression" in self.thresholds:
            reg_passed, reg_failures, reg_details = self._validate_regression(
                metrics, self.thresholds["regression"], self.thresholds.get("baseline_file")
            )
            passed = passed and reg_passed
            failures.extend(reg_failures)
            details["regression"] = reg_details

        return passed, failures, details

    def _validate_absolute(self, metrics: Dict[str, Any], thresholds: Dict[str, float]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate absolute thresholds."""
        passed = True
        failures = []
        details = {}

        for metric_path, threshold in thresholds.items():
            value = self._get_nested_value(metrics, metric_path)
            if value is None:
                passed = False
                failures.append(f"Metric {metric_path} not found")
                details[metric_path] = {"passed": False, "reason": "metric not found"}
                continue

            metric_passed = value <= threshold
            if not metric_passed:
                passed = False
                failures.append(f"Metric {metric_path}: {value} > {threshold}")

            details[metric_path] = {
                "passed": metric_passed,
                "value": value,
                "threshold": threshold
            }

        return passed, failures, details

    def _validate_regression(self, metrics: Dict[str, Any], thresholds: Dict[str, float], baseline_file: Optional[str]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate regression thresholds."""
        if not baseline_file:
            return True, [], {}

        try:
            with open(baseline_file) as f:
                baseline = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return False, [f"Failed to load baseline: {e}"], {}

        passed = True
        failures = []
        details = {}

        for metric_path, max_regression in thresholds.items():
            current_value = self._get_nested_value(metrics, metric_path)
            baseline_value = self._get_nested_value(baseline, metric_path)

            if current_value is None or baseline_value is None:
                passed = False
                failures.append(f"Metric {metric_path} not found in current or baseline")
                details[metric_path] = {"passed": False, "reason": "metric not found"}
                continue

            if baseline_value == 0:
                regression_percent = 0 if current_value == 0 else float('inf')
            else:
                regression_percent = ((current_value - baseline_value) / baseline_value) * 100

            metric_passed = regression_percent <= max_regression
            if not metric_passed:
                passed = False
                failures.append(f"Metric {metric_path}: {regression_percent:.2f}% regression > {max_regression}%")

            details[metric_path] = {
                "passed": metric_passed,
                "current_value": current_value,
                "baseline_value": baseline_value,
                "regression_percent": regression_percent,
                "max_regression": max_regression
            }

        return passed, failures, details

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Optional[Any]:
        """Get nested value using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current

    @staticmethod
    def save_baseline(metrics: Dict[str, Any], baseline_file: Path) -> None:
        """Save metrics as baseline for future regression testing."""
        with baseline_file.open('w') as f:
            json.dump(metrics, f, indent=2)