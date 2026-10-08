"""Deterministic, aggregate-only merchant analytics."""

from importlib import import_module

__all__ = [
    "get_sales_performance", "get_day_of_week_analysis", "get_daypart_analysis",
    "get_repeat_customer_analysis", "get_lapsed_customer_summary",
    "get_peer_benchmark", "get_merchant_facts", "get_weekly_sales_pulse", "get_sales_overview",
]

_MODULES = {
    "get_sales_performance": "performance",
    "get_weekly_sales_pulse": "performance",
    "get_sales_overview": "overview",
    "get_day_of_week_analysis": "performance",
    "get_daypart_analysis": "performance",
    "get_repeat_customer_analysis": "customers",
    "get_lapsed_customer_summary": "customers",
    "get_peer_benchmark": "peers",
    "get_merchant_facts": "facts",
}


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(name)
    return getattr(import_module(f"sme_growth.analytics.{_MODULES[name]}"), name)
