"""
Base Agent Abstract Class for the Multi-Agent Prediction Team.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import time
from datetime import datetime
from core.schema import AgentExecutionReport

class BaseAgent(ABC):
    def __init__(self, name: str, role: str, criteria: str):
        self.name = name
        self.role = role
        self.criteria = criteria

    def describe(self) -> str:
        return f"[{self.name}] - Role: {self.role} | Criteria: {self.criteria}"

    def run(self, **kwargs) -> tuple[Any, AgentExecutionReport]:
        """
        Executes the agent's task with automatic timing, safety isolation, and execution reporting.
        """
        start_time = time.time()
        try:
            result = self.execute(**kwargs)
            elapsed = time.time() - start_time
            report = AgentExecutionReport(
                agent_name=self.name,
                role=self.role,
                success=True,
                execution_time_seconds=round(elapsed, 3),
                message=f"Successfully analyzed in {round(elapsed, 2)}s"
            )
            return result, report
        except Exception as e:
            elapsed = time.time() - start_time
            report = AgentExecutionReport(
                agent_name=self.name,
                role=self.role,
                success=False,
                execution_time_seconds=round(elapsed, 3),
                message=f"Error executing agent {self.name}: {str(e)}"
            )
            raise e

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Core logic to be implemented by each specialized agent."""
        pass
