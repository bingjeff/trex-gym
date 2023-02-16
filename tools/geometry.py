# Basic datastructures for describing elements of URDF-like files.
import dataclasses
import numpy as np

from scipy.spatial import transform


@dataclasses.dataclass
class Transform:
    translation: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )
    rotation: transform.Rotation = dataclasses.field(
        default_factory=transform.Rotation.identity
    )

    def __mul__(self, rhs: "Transform") -> "Transform":
        return Transform(
            translation=self.apply(rhs.translation),
            rotation=self.rotation * rhs.rotation,
        )

    def inverse(self) -> "Transform":
        inv_rotation = self.rotation.inv()
        return Transform(
            translation=-inv_rotation.apply(self.translation),
            rotation=inv_rotation,
        )

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        if len(vectors) == 3:
            return (
                (self.rotation.as_matrix() @ vectors).T + self.translation
            ).T
        else:
            return (self.rotation.as_matrix() @ vectors.T).T + self.translation


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
    origin: Transform = Transform()


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


@dataclasses.dataclass
class GeometryBox(Geometry):
    size_xyz: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0])
    )

    @property
    def length_x(self) -> float:
        return self.size_xyz[0]

    @property
    def length_y(self) -> float:
        return self.size_xyz[1]

    @property
    def length_z(self) -> float:
        return self.size_xyz[2]
