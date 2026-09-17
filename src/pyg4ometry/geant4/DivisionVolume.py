from .PhysicalVolume import PhysicalVolume as _PhysicalVolume
from . import solid as _solid
from ..visualisation import Mesh as _Mesh
from ..visualisation import VisualisationOptions as _VisOptions
from .. import transformation as _trans

import numpy as _np
import copy as _copy
import logging as _log

_log = _log.getLogger(__name__)


class DivisionVolume(_PhysicalVolume):
    """
    DivisionVolume: G4PVDivision

    :param name: of physical volume
    :param logical: volume to be placed
    :param mother: logical volume,
    :param axis: kXAxis,kYAxis,kZAxis,kRho,kPhi
    :param ncopies: number of replicas
    :param width: spacing between replicas along axis
    :param offset: of grid
    """

    class Axis:
        kXAxis = 1
        kYAxis = 2
        kZAxis = 3
        kRho = 4
        kPhi = 5

    def __init__(
        self,
        name,
        logicalVolume,
        motherVolume,
        axis,
        ndivisions=-1,
        width=-1,
        offset=0,
        registry=None,
        addRegistry=True,
        unit="mm",
    ):
        self.type = "division"
        self.name = name
        self.logicalVolume = logicalVolume
        self.motherVolume = motherVolume
        self.motherVolume.add(self)
        self.axis = axis
        self.ndivisions = ndivisions
        self.width = width
        self.offset = offset
        self.unit = unit

        self.visOptions = None

        # NOT PART OF NORMAL DIVISION VOLUME BUT USEFUL FOR CONVERSION TO FLUKA
        # need to determine type or rotation and position, as should be Position or Rotation type
        from ..gdml import Defines as _Defines

        self.position = _Defines.Position(name + "_pos", 0, 0, 0, "mm", registry, False)
        self.rotation = _Defines.Rotation(name + "_rot", 0, 0, 0, "rad", registry, False)
        self.scale = _Defines.Scale(name + "_sca", 1, 1, 1, "none", registry, False)

        if motherVolume.solid.type != logicalVolume.solid.type:
            msg = f"Can not have divisions with a different solid type than the mother volume. Mother : {motherVolume.solid.type}, Division : {logicalVolume.solid.type}"
            raise ValueError(msg)
        if addRegistry:
            registry.addPhysicalVolume(self)

        # physical visualisation options
        self.visOptions = _VisOptions()

        # Create division meshes
        [self.meshes, self.transforms] = self.createDivisionMeshes()

    def getMotherSize(self):
        sd = self.motherVolume.solid
        stype = sd.type

        # The sizes along the 5 axis for all supported solids
        if stype == "Box":
            sizes = [float(sd.pX), float(sd.pY), float(sd.pZ), -1, -1]

        elif stype == "Tubs":
            sizes = [
                -1,
                -1,
                float(sd.pDz),
                float(sd.pRMax) - float(sd.pRMin),
                float(sd.pDPhi),
            ]

        elif stype == "Cons":
            # The radius is the outer radius at the -Z face
            sizes = [
                -1,
                -1,
                float(sd.pDz),
                float(sd.pRmax1) - float(sd.pRmin1),
                float(sd.pDPhi),
            ]

        elif stype == "Trd":
            # Can not divide up the sloping sides of the trapezoid
            sizes = [
                min(float(sd.pX1), float(sd.pX2)),
                min(float(sd.pY1), float(sd.pY2)),
                float(sd.pZ),
                -1,
                -1,
            ]

        elif stype == "Para":
            sizes = [2 * float(sd.pX), 2 * float(sd.pY), 2 * float(sd.pZ), -1, -1]

        elif stype == "Polycone":
            # Z is in increasing order
            sizes = [
                -1,
                -1,
                float(sd.pZpl[-1]) - float(sd.pZpl[0]),
                float(sd.pRMax[0]) - float(sd.pRMin[0]),
                float(sd.pDPhi),
            ]

        elif stype == "Polyhedra":
            sizes = [
                -1,
                -1,
                float(sd.zPlane[-1]) - float(sd.zPlane[0]),
                float(sd.rOuter[0]) - float(sd.rInner[0]),
                float(sd.pDPhi),
            ]

        return sizes[self.axis - 1]

    def checkAxis(self, allowed_axes):
        if self.axis not in allowed_axes:
            msg = f"Division along axis {self.axis} not supported for solid {self.logicalVolume.solid.type}"
            raise ValueError(msg)

    def divideBox(self, offset, width, ndiv):
        allowed_axes = [self.Axis.kXAxis, self.Axis.kYAxis, self.Axis.kZAxis]
        self.checkAxis(allowed_axes)

        meshes = []
        transforms = []

        msize = self.getMotherSize()
        placements = _np.arange(
            -msize / 2.0 + offset + width / 2.0,
            -msize / 2.0 + offset + width * ndiv,
            width,
        )

        for i, v in enumerate(placements):
            # slice parameter values are computed up front and passed as plain
            # numbers - the mother solid's parameters must never be mutated
            pX = float(self.motherVolume.solid.pX)
            pY = float(self.motherVolume.solid.pY)
            pZ = float(self.motherVolume.solid.pZ)

            if self.axis == self.Axis.kXAxis:
                pX = width
                transforms.append([[0, 0, 0], [v, 0, 0]])

            elif self.axis == self.Axis.kYAxis:
                pY = width
                transforms.append([[0, 0, 0], [0, v, 0]])

            elif self.axis == self.Axis.kZAxis:
                pZ = width
                transforms.append([[0, 0, 0], [0, 0, v]])

            solid = _solid.Box(
                self.name + "_" + self.motherVolume.solid.name + "_" + str(i),
                pX,
                pY,
                pZ,
                self.logicalVolume.registry,
                self.motherVolume.solid.lunit,
                False,
            )

            meshes.append(_Mesh(solid))
        return meshes, transforms

    def divideTubs(self, offset, width, ndiv):
        allowed_axes = [self.Axis.kRho, self.Axis.kPhi, self.Axis.kZAxis]
        self.checkAxis(allowed_axes)

        meshes = []
        transforms = []

        if self.axis == self.Axis.kPhi:
            # Always take into account the inner sizes of the solids
            placements = _np.arange(
                float(self.motherVolume.solid.pSPhi) + offset,
                float(self.motherVolume.solid.pSPhi) + offset + ndiv * width,
                width,
            )
        elif self.axis == self.Axis.kRho:
            placements = _np.arange(
                float(self.motherVolume.solid.pRMin) + offset,
                float(self.motherVolume.solid.pRMin) + offset + ndiv * width,
                width,
            )
        else:  # Position axes
            msize = self.getMotherSize()
            placements = _np.arange(
                -msize / 2.0 + offset + width / 2.0,
                -msize / 2.0 + offset + width * ndiv,
                width,
            )

        for i, v in enumerate(placements):
            pRMin = float(self.motherVolume.solid.pRMin)
            pRMax = float(self.motherVolume.solid.pRMax)
            pDz = float(self.motherVolume.solid.pDz)
            pSPhi = float(self.motherVolume.solid.pSPhi)
            pDPhi = float(self.motherVolume.solid.pDPhi)

            if self.axis == self.Axis.kZAxis:
                pDz = width
                transforms.append([[0, 0, 0], [0, 0, v]])

            elif self.axis == self.Axis.kRho:
                pRMin = v
                pRMax = v + width
                transforms.append([[0, 0, 0], [0, 0, 0]])

            elif self.axis == self.Axis.kPhi:
                pSPhi = v
                pDPhi = width
                transforms.append([[0, 0, 0], [0, 0, 0]])

            solid = _solid.Tubs(
                self.name + "_" + self.motherVolume.solid.name + "_" + str(i),
                pRMin,
                pRMax,
                pDz,
                pSPhi,
                pDPhi,
                self.motherVolume.registry,
                self.motherVolume.solid.lunit,
                self.motherVolume.solid.aunit,
                self.logicalVolume.solid.nslice,
                False,
            )

            meshes.append(_Mesh(solid))

        return meshes, transforms

    def divideCons(self, offset, width, ndiv):
        allowed_axes = [self.Axis.kRho, self.Axis.kPhi, self.Axis.kZAxis]
        self.checkAxis(allowed_axes)

        meshes = []
        transforms = []

        msize = self.getMotherSize()
        if self.axis == self.Axis.kPhi:
            # Always take into account the inner sizes of the solids
            placements = _np.arange(
                float(self.motherVolume.solid.pSPhi) + offset,
                float(self.motherVolume.solid.pSPhi) + offset + ndiv * width,
                width,
            )
        elif self.axis == self.Axis.kRho:
            placements = _np.arange(
                float(self.motherVolume.solid.pRmin1) + offset,
                float(self.motherVolume.solid.pRmin1) + offset + ndiv * width,
                width,
            )
        else:  # Position axes
            placements = _np.arange(
                -msize / 2.0 + offset + width / 2.0,
                -msize / 2.0 + offset + width * ndiv,
                width,
            )

        r1 = float(self.motherVolume.solid.pRmin1)
        r2 = float(self.motherVolume.solid.pRmin2)
        R1 = float(self.motherVolume.solid.pRmax1)
        R2 = float(self.motherVolume.solid.pRmax2)
        dr = r2 - r1
        dR = R2 - R1
        w_ratio = (R2 - r2) / msize  # Ratio of the bottom and top thickness

        h_i = 0.0  # For linear interpolation of the Z-divisions
        r_i = r1
        R_i = R1
        for i, v in enumerate(placements):
            pRmin1 = float(self.motherVolume.solid.pRmin1)
            pRmax1 = float(self.motherVolume.solid.pRmax1)
            pRmin2 = float(self.motherVolume.solid.pRmin2)
            pRmax2 = float(self.motherVolume.solid.pRmax2)
            pDz = float(self.motherVolume.solid.pDz)
            pSPhi = float(self.motherVolume.solid.pSPhi)
            pDPhi = float(self.motherVolume.solid.pDPhi)

            if self.axis == self.Axis.kZAxis:
                pRmin1 = r_i
                pRmax1 = R_i
                h_i += width
                r_i = r1 + h_i * dr / msize
                R_i = R1 + h_i * dR / msize
                pRmin2 = r_i
                pRmax2 = R_i
                pDz = width
                transforms.append([[0, 0, 0], [0, 0, v]])

            elif self.axis == self.Axis.kRho:
                pRmin1 = v
                pRmax1 = v + width
                v_2 = (
                    v - (r1 + offset) + (r2 + w_ratio * offset)
                )  # Transfrom to top starting offset
                pRmin2 = v_2
                pRmax2 = v_2 + w_ratio * width
                transforms.append([[0, 0, 0], [0, 0, 0]])

            elif self.axis == self.Axis.kPhi:
                pSPhi = v
                pDPhi = width
                transforms.append([[0, 0, 0], [0, 0, 0]])

            solid = _solid.Cons(
                self.name + "_" + self.motherVolume.solid.name + "_" + str(i),
                pRmin1,
                pRmax1,
                pRmin2,
                pRmax2,
                pDz,
                pSPhi,
                pDPhi,
                self.motherVolume.registry,
                self.motherVolume.solid.lunit,
                self.motherVolume.solid.aunit,
                self.motherVolume.solid.nslice,
                False,
            )

            meshes.append(_Mesh(solid))

        return meshes, transforms

    def dividePara(self, offset, width, ndiv):
        allowed_axes = [self.Axis.kXAxis, self.Axis.kYAxis, self.Axis.kZAxis]
        self.checkAxis(allowed_axes)

        meshes = []
        transforms = []

        msize = self.getMotherSize()
        placements = _np.arange(
            -msize / 2.0 + offset + width / 2.0,
            -msize / 2.0 + offset + width * ndiv,
            width,
        )

        for i, v in enumerate(placements):
            pX = float(self.motherVolume.solid.pX)
            pY = float(self.motherVolume.solid.pY)
            pZ = float(self.motherVolume.solid.pZ)
            pAlpha = float(self.motherVolume.solid.pAlpha)
            pTheta = float(self.motherVolume.solid.pTheta)
            pPhi = float(self.motherVolume.solid.pPhi)

            if self.axis == self.Axis.kXAxis:
                pX = width / 2.0
                transforms.append([[0, 0, 0], [v, 0, 0]])

            elif self.axis == self.Axis.kYAxis:
                pY = width / 2.0
                transforms.append([[0, 0, 0], [v * _np.sin(pAlpha), v, 0]])

            elif self.axis == self.Axis.kZAxis:
                pZ = width / 2.0
                transforms.append(
                    [
                        [0, 0, 0],
                        [v * _np.sin(pTheta), v * _np.sin(pPhi), v * _np.cos(pPhi)],
                    ]
                )

            solid = _solid.Para(
                self.name + "_" + self.motherVolume.solid.name + "_" + str(i),
                pX,
                pY,
                pZ,
                pAlpha,
                pTheta,
                pPhi,
                self.motherVolume.registry,
                self.motherVolume.solid.lunit,
                self.motherVolume.solid.aunit,
                False,
            )

            meshes.append(_Mesh(solid))

        return meshes, transforms

    def divideTrd(self, offset, width, ndiv):
        allowed_axes = [self.Axis.kXAxis, self.Axis.kYAxis, self.Axis.kZAxis]
        self.checkAxis(allowed_axes)

        meshes = []
        transforms = []

        msize = self.getMotherSize()
        placements = _np.arange(
            -msize / 2.0 + offset + width / 2.0,
            -msize / 2.0 + offset + width * ndiv,
            width,
        )

        x1 = float(self.motherVolume.solid.pX1)
        x2 = float(self.motherVolume.solid.pX2)
        y1 = float(self.motherVolume.solid.pY1)
        y2 = float(self.motherVolume.solid.pY2)

        dX = x2 - x1
        dY = y2 - y1
        x_i = x1  # For linear interpolation in Z
        y_i = y1
        h_i = 0.0

        for i, v in enumerate(placements):
            pX1 = float(self.motherVolume.solid.pX1)
            pX2 = float(self.motherVolume.solid.pX2)
            pY1 = float(self.motherVolume.solid.pY1)
            pY2 = float(self.motherVolume.solid.pY2)
            pZ = float(self.motherVolume.solid.pZ)

            if self.axis == self.Axis.kXAxis:
                pX1 = width
                pX2 = width
                transforms.append([[0, 0, 0], [v, 0, 0]])

            elif self.axis == self.Axis.kYAxis:
                pY1 = width
                pY2 = width
                transforms.append([[0, 0, 0], [0, v, 0]])

            elif self.axis == self.Axis.kZAxis:
                pX1 = x_i
                pY1 = y_i
                h_i += width
                x_i = x1 + h_i * dX / msize
                y_i = y1 + h_i * dY / msize
                pX2 = x_i
                pY2 = y_i
                pZ = width
                transforms.append([[0, 0, 0], [0, 0, v]])

            solid = _solid.Trd(
                self.name + "_" + self.motherVolume.solid.name + "_" + str(i),
                pX1,
                pX2,
                pY1,
                pY2,
                pZ,
                self.motherVolume.registry,
                self.motherVolume.solid.lunit,
                False,
            )

            meshes.append(_Mesh(solid))

        return meshes, transforms

    def dividePolycone(self, offset, width, ndiv):
        allowed_axes = [self.Axis.kRho, self.Axis.kPhi, self.Axis.kZAxis]
        self.checkAxis(allowed_axes)

        meshes = []
        transforms = []

        msize = self.getMotherSize()
        if not msize:  # Possible if inner and outer radii at -Z are the same
            msg = "Cannot construct polycone division with degenerate radii at -Z"
            raise ValueError(msg)

        if self.axis == self.Axis.kPhi:
            # Always take into account the inner sizes of the solids
            placements = _np.arange(
                float(self.motherVolume.solid.pSPhi) + offset,
                float(self.motherVolume.solid.pSPhi) + offset + ndiv * width,
                width,
            )
        elif self.axis == self.Axis.kRho:
            placements = _np.arange(
                float(self.motherVolume.solid.pRMin[0]) + offset,
                float(self.motherVolume.solid.pRMin[0]) + offset + ndiv * width,
                width,
            )
        else:
            # If width is not specified, divide along the Z planes
            # If the width is specified, can only divide between 2 Z planes
            if ndiv * width == msize:  # this means default width
                placements = _np.array([float(t) for t in self.motherVolume.solid.pZpl])
            else:
                zpl_sizes = _np.diff([float(t) for t in self.motherVolume.solid.pZpl])
                zsl_index = 0
                zsl_size = 0
                offs = offset
                for i, zs in enumerate(zpl_sizes):
                    offs -= zs
                    if offs < 0:
                        if ndiv * width > abs(offs):
                            msg = "Division with user-specified width is only possible between 2 z-planes."
                            raise ValueError(msg)
                        zsl_index = i
                        zsl_size = zs
                        break

                placements = _np.arange(
                    float(self.motherVolume.solid.pZpl[0]) + offset,
                    float(self.motherVolume.solid.pZpl[0]) + offset + width * ndiv,
                    width,
                )

                z_1 = float(self.motherVolume.solid.pZpl[zsl_index])
                z_2 = float(self.motherVolume.solid.pZpl[zsl_index + 1])
                r_1 = float(self.motherVolume.solid.pRMin[zsl_index])
                r_2 = float(self.motherVolume.solid.pRMin[zsl_index + 1])
                R_1 = float(self.motherVolume.solid.pRMax[zsl_index])
                R_2 = float(self.motherVolume.solid.pRMax[zsl_index + 1])

                dr = r_2 - r_1
                dR = R_2 - R_1
                dz = z_2 - z_1
                h_i = offset - sum(zpl_sizes[:zsl_index])

        for i, v in enumerate(placements):
            pSPhi = float(self.motherVolume.solid.pSPhi)
            pDPhi = float(self.motherVolume.solid.pDPhi)
            pZpl = [float(t) for t in self.motherVolume.solid.pZpl]
            pRMin = [float(t) for t in self.motherVolume.solid.pRMin]
            pRMax = [float(t) for t in self.motherVolume.solid.pRMax]

            if self.axis == self.Axis.kRho:
                r_0 = pRMin[0]
                w_0 = pRMax[0] - r_0
                pRMinNew = []
                pRMaxNew = []
                for j in range(len(pRMin)):
                    r_j = pRMin[j]
                    w_j = pRMax[j] - r_j
                    w_ratio_j = w_j / w_0
                    v_2 = w_ratio_j * (v - (r_0 + offset)) + (
                        r_j + w_ratio_j * offset
                    )  # Proprtional increase
                    pRMinNew.append(v_2)
                    pRMaxNew.append(v_2 + w_ratio_j * width)
                pRMin = pRMinNew
                pRMax = pRMaxNew
                transforms.append([[0, 0, 0], [0, 0, 0]])

            elif self.axis == self.Axis.kPhi:
                pSPhi = v
                pDPhi = width
                transforms.append([[0, 0, 0], [0, 0, 0]])

            elif self.axis == self.Axis.kZAxis:
                if ndiv * width == msize:
                    # This is the default case and we don't actually need the calculated
                    # placements, only the indices
                    if i == len(pRMin) - 1:
                        continue  # As we split into (nzplanes - 1) polycones
                    pRMin = pRMin[i : i + 2]
                    pRMax = pRMax[i : i + 2]
                    pZpl = pZpl[i : i + 2]
                else:
                    r_min = []
                    r_max = []
                    r_min.append(r_1 + h_i * dr / dz)
                    r_max.append(R_1 + h_i * dR / dz)
                    h_i += width
                    r_min.append(r_1 + h_i * dr / dz)
                    r_max.append(R_1 + h_i * dR / dz)
                    pRMin = r_min
                    pRMax = r_max
                    pZpl = [v, v + width]
                    transforms.append([[0, 0, 0], [0, 0, 0]])

            solid = _solid.Polycone(
                self.name + "_" + self.motherVolume.solid.name + "_" + str(i),
                pSPhi,
                pDPhi,
                pZpl,
                pRMin,
                pRMax,
                self.motherVolume.registry,
                self.motherVolume.solid.lunit,
                self.motherVolume.solid.aunit,
                self.motherVolume.solid.nslice,
                False,
            )

            meshes.append(_Mesh(solid))

        return meshes, transforms

    def dividePolyhedra(self, offset, width, ndiv):
        allowed_axes = [self.Axis.kRho, self.Axis.kPhi, self.Axis.kZAxis]
        self.checkAxis(allowed_axes)

        meshes = []
        transforms = []

        msize = self.getMotherSize()
        if not msize:  # Possible if inner and outer radii at -Z are the same
            msg = "Cannot construct polyhedra division with degenerate radii at -Z"
            raise ValueError(msg)

        if self.axis == self.Axis.kPhi:
            # Always take into account the inner sizes of the solids
            sphi = float(self.motherVolume.solid.pSPhi)
            dphi = float(self.motherVolume.solid.pDPhi)
            nsides = int(float(self.motherVolume.solid.numSide))
            placements = _np.arange(sphi, sphi + dphi, dphi / nsides)

        elif self.axis == self.Axis.kRho:
            placements = _np.arange(
                float(self.motherVolume.solid.rInner[0]) + offset,
                float(self.motherVolume.solid.rInner[0]) + offset + ndiv * width,
                width,
            )
        else:
            # If width is not specified, divide along the Z planes
            # If the width is specified, can only divide between 2 Z planes
            if ndiv * width == msize:  # this means default width
                placements = _np.array([float(t) for t in self.motherVolume.solid.zPlane])
            else:
                zpl_sizes = _np.diff([float(t) for t in self.motherVolume.solid.zPlane])
                zsl_index = 0
                zsl_size = 0
                offs = offset
                for i, zs in enumerate(zpl_sizes):
                    offs -= zs
                    if offs < 0:
                        if ndiv * width > abs(offs):
                            msg = "Division with user-specified width is only possible between 2 z-planes."
                            raise ValueError(msg)
                        zsl_index = i
                        zsl_size = zs
                        break

                placements = _np.arange(
                    float(self.motherVolume.solid.zPlane[0]) + offset,
                    float(self.motherVolume.solid.zPlane[0]) + offset + width * ndiv,
                    width,
                )

                z_1 = float(self.motherVolume.solid.zPlane[zsl_index])
                z_2 = float(self.motherVolume.solid.zPlane[zsl_index + 1])
                r_1 = float(self.motherVolume.solid.rInner[zsl_index])
                r_2 = float(self.motherVolume.solid.rInner[zsl_index + 1])
                R_1 = float(self.motherVolume.solid.rOuter[zsl_index])
                R_2 = float(self.motherVolume.solid.rOuter[zsl_index + 1])

                dr = r_2 - r_1
                dR = R_2 - R_1
                dz = z_2 - z_1
                h_i = offset - sum(zpl_sizes[:zsl_index])

        for i, v in enumerate(placements):
            pSPhi = float(self.motherVolume.solid.pSPhi)
            pDPhi = float(self.motherVolume.solid.pDPhi)
            numSide = float(self.motherVolume.solid.numSide)
            zPlane = [float(t) for t in self.motherVolume.solid.zPlane]
            rInner = [float(t) for t in self.motherVolume.solid.rInner]
            rOuter = [float(t) for t in self.motherVolume.solid.rOuter]

            if self.axis == self.Axis.kRho:
                r_0 = rInner[0]
                w_0 = rOuter[0] - r_0
                rInnerNew = []
                rOuterNew = []
                for j in range(len(rInner)):
                    r_j = rInner[j]
                    w_j = rOuter[j] - r_j
                    w_ratio_j = w_j / w_0
                    v_2 = w_ratio_j * (v - (r_0 + offset)) + (
                        r_j + w_ratio_j * offset
                    )  # Proprtional increase
                    rInnerNew.append(v_2)
                    rOuterNew.append(v_2 + w_ratio_j * width)
                rInner = rInnerNew
                rOuter = rOuterNew
                transforms.append([[0, 0, 0], [0, 0, 0]])

            elif self.axis == self.Axis.kPhi:
                pSPhi = v
                pDPhi = dphi / nsides
                numSide = 1
                transforms.append([[0, 0, 0], [0, 0, 0]])

            elif self.axis == self.Axis.kZAxis:
                if ndiv * width == msize:
                    # This is the default case and we don't actually need the calculated
                    # placements, only the indices
                    if i == len(rInner) - 1:
                        continue  # As we split into (nzplanes - 1) polycones
                    rInner = rInner[i : i + 2]
                    rOuter = rOuter[i : i + 2]
                    zPlane = zPlane[i : i + 2]
                else:
                    r_min = []
                    r_max = []
                    r_min.append(r_1 + h_i * dr / dz)
                    r_max.append(R_1 + h_i * dR / dz)
                    h_i += width
                    r_min.append(r_1 + h_i * dr / dz)
                    r_max.append(R_1 + h_i * dR / dz)
                    rInner = r_min
                    rOuter = r_max
                    zPlane = [v, v + width]
                    transforms.append([[0, 0, 0], [0, 0, 0]])

            solid = _solid.Polyhedra(
                self.name + "_" + self.motherVolume.solid.name + "_" + str(i),
                pSPhi,
                pDPhi,
                numSide,
                self.motherVolume.solid.numZPlanes,
                zPlane,
                rInner,
                rOuter,
                self.motherVolume.registry,
                self.motherVolume.solid.lunit,
                self.motherVolume.solid.aunit,
                False,
            )

            meshes.append(_Mesh(solid))

        return meshes, transforms

    def createDivisionMeshes(self):
        ndivisions = int(
            float(self.ndivisions)
        )  # Do float() instead of .eval() because .eval() doesnt
        offset = float(self.offset)  # work with the default numerical values
        width = float(self.width)

        transforms = []
        meshes = []

        # Poor man's overloading of the 3 possible constructors.
        if width <= 0 and ndivisions > 0:
            # raise ValueError("Option not implemented yet")
            width = (self.getMotherSize() - offset) / ndivisions
        elif ndivisions <= 0 and width > 0:
            # raise ValueError("Option not implemented yet")
            width = int((self.getMotherSize() - offset) / width)
        elif ndivisions > 0 and width > 0:
            pass  # Can work with this directly

        if hasattr(self, f"divide{self.logicalVolume.solid.type}"):
            stype = self.logicalVolume.solid.type
            meshes, transforms = getattr(self, f"divide{stype}")(offset, width, ndivisions)
        else:
            msg = f"Division with solid {self.logicalVolume.solid.type} is not supported yet."
            raise ValueError(msg)

        return [meshes, transforms]

    def __repr__(self):
        return f"Division volume : {self.name} {self.axis} {self.ndivisions} {self.offset} {self.width}"

    def extent(self, includeBoundingSolid=True):
        _log.debug(f"ReplicaVolume.extent> {self.name}")

        vMin = [1e99, 1e99, 1e99]
        vMax = [-1e99, -1e99, -1e99]

        for trans, mesh in zip(self.transforms, self.meshes):
            # transform daughter meshes to parent coordinates
            dvmrot = _trans.tbxyz2matrix(trans[0])
            dvtra = _np.array(trans[1])

            [vMinDaughter, vMaxDaughter] = mesh.getBoundingBox()

            vMinDaughter = _np.array(dvmrot.dot(vMinDaughter) + dvtra).flatten()
            vMaxDaughter = _np.array(dvmrot.dot(vMaxDaughter) + dvtra).flatten()

            if vMaxDaughter[0] > vMax[0]:
                vMax[0] = vMaxDaughter[0]
            if vMaxDaughter[1] > vMax[1]:
                vMax[1] = vMaxDaughter[1]
            if vMaxDaughter[2] > vMax[2]:
                vMax[2] = vMaxDaughter[2]

            if vMinDaughter[0] < vMin[0]:
                vMin[0] = vMinDaughter[0]
            if vMinDaughter[1] < vMin[1]:
                vMin[1] = vMinDaughter[1]
            if vMinDaughter[2] < vMin[2]:
                vMin[2] = vMinDaughter[2]

        return [vMin, vMax]
