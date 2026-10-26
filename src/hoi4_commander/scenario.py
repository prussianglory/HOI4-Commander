"""Конечный сценарий мирной экономики: Германия, 1 января 1936 года."""
from .models import Step


def germany_1936() -> tuple[Step, ...]:
    rows = [
        ("open_focus", "focus_menu", "focus_available", "Открываю национальные фокусы для выбора экономического направления."),
        ("choose_focus", "focus_available", "focus_start", "Выбираю доступный экономический фокус из откалиброванного набора."),
        ("start_focus", "focus_start", "focus_active", "Подтверждаю фокус: он должен стать активным."),
        ("open_research", "research_menu", "research_slot", "Открываю исследования, чтобы занять свободную ячейку."),
        ("choose_slot", "research_slot", "technology_available", "Открываю свободную исследовательскую ячейку."),
        ("choose_technology", "technology_available", "technology_start", "Выбираю доступную промышленную технологию из откалиброванного набора."),
        ("start_technology", "technology_start", "research_active", "Подтверждаю исследование и проверяю назначение технологии."),
        ("open_construction", "construction_menu", "civilian_factory_tool", "Открываю строительство для развития гражданской промышленности."),
        ("choose_civilian", "civilian_factory_tool", "building_state", "Выбираю инструмент строительства гражданской фабрики."),
        ("queue_factory", "building_state", "construction_queued", "Добавляю одну гражданскую фабрику в подходящем регионе и проверяю очередь."),
    ]
    return tuple(Step(id=i, target_id=t, expected_present=(e,), explanation=r) for i, t, e, r in rows)
