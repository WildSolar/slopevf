import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from slopevf import Array  # noqa: E402

DATA = HERE / "data"

# Geometries used throughout the tests: example_script.m, a steep vertical array,
# horizontal ground, and a dense low array (strong inhomogeneity, paper Fig. 15).
GEOMETRIES = {
    "example": Array(60, 20, 1.125, 1.84, 0),
    "vertical_steep": Array(90, 30, 0.9, 1.2, -25),
    "horizontal": Array(30, 0, 2.5, 1.0, 15),
    "low_dense": Array.from_shading_angle(60, 20, 15, 0.4),
    # panels of finite thickness (module stacks)
    "example_thick": Array(60, 20, 1.125, 1.84, 0, thickness=0.16),
    "totalp_thick": Array(90, 20, 1.126, 1.884, 0, thickness=15 / 96),
    "tilted_thick": Array(30, 0, 2.5, 1.0, 15, thickness=0.08),
}


@pytest.fixture(params=list(GEOMETRIES), ids=list(GEOMETRIES))
def array(request):
    return GEOMETRIES[request.param]
