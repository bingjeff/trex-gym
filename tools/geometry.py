# Basic datastructures for describing elements of URDF-like files.
import dataclasses
import numpy as np

from scipy.spatial import transform


@dataclasses.dataclass
class Frame:
    translation: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    rotation: transform.Rotation = dataclasses.field(
        default_factory=transform.Rotation.identity
    )


def _inf_bounds():
    return np.array([-np.inf, np.inf])


@dataclasses.dataclass
class MotionLimits:
    position: np.ndarray = dataclasses.field(default_factory=_inf_bounds)
    velocity: np.ndarray = dataclasses.field(default_factory=_inf_bounds)
    acceleration: np.ndarray = dataclasses.field(default_factory=_inf_bounds)


@dataclasses.dataclass
class Geometry:
    _: dataclasses.KW_ONLY
    origin: Frame = Frame()


@dataclasses.dataclass
class GeometryMesh(Geometry):
    filename: str


@dataclasses.dataclass
class GeometrySphere(Geometry):
    radius: float


@dataclasses.dataclass
class GeometryCapsule(Geometry):
    radius: float
    length: float
