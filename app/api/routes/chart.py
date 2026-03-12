"""KALESS Engine — Chart Route (chart data generation)."""

from fastapi import APIRouter, Depends

from app.api.deps import verify_engine_key

router = APIRouter()


@router.post("/chart")
async def generate_chart_data(
    _key: str = Depends(verify_engine_key),
):
    """Generate chart-ready data for visualization.

    Returns data formatted for Recharts consumption.
    """
    # TODO: Implement in Phase 5
    return {"status": "not_implemented"}
