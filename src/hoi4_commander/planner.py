"""Планировщик допускает только действие текущего шага сценария."""
from .model_provider import ModelProvider, ProviderError, validate_choice
from .models import Observation, Plan, Step


class PlanningError(RuntimeError):
    """План не может быть безопасно исполнен."""


class ScenarioPlanner:
    def __init__(self, provider: ModelProvider):
        self.provider = provider

    def plan(self, observation: Observation, step: Step) -> Plan:
        try:
            plan = self.provider.choose(observation, step)
            if not isinstance(plan, Plan):
                raise PlanningError("Провайдер не вернул типизированный план")
            return validate_choice(plan, observation, step)
        except ProviderError as exc:
            raise PlanningError(str(exc)) from None
