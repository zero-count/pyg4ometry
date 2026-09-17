import numpy as _np

import pyg4ometry.gdml as _gd
import pyg4ometry.geant4 as _g4


def _registry_with(solid_mother, solid_daughter):
    reg = _g4.Registry()
    wm = _g4.MaterialPredefined("G4_Galactic")
    bm = _g4.MaterialPredefined("G4_Fe")
    ml = _g4.LogicalVolume(solid_mother, wm, "ml", reg)
    dl = _g4.LogicalVolume(solid_daughter, bm, "dl", reg)
    return reg, ml, dl


def test_DivisionMeshesDoNotMutateMotherSolidBox():
    """Building the slice meshes of a DivisionVolume must not change the
    mother solid's parameters (they were shrunk to the slice width by
    in-place expression edits)."""
    reg = _g4.Registry()
    mbx = _gd.Constant("mbx", "100", reg, True)
    mby = _gd.Constant("mby", "100", reg, True)
    mbz = _gd.Constant("mbz", "100", reg, True)

    ms = _g4.solid.Box("ms", mbx, mby, mbz, reg, "mm")
    ds = _g4.solid.Box("ds", 20, 100, 100, reg, "mm")

    wm = _g4.MaterialPredefined("G4_Galactic")
    bm = _g4.MaterialPredefined("G4_Fe")
    ml = _g4.LogicalVolume(ms, wm, "ml", reg)
    dl = _g4.LogicalVolume(ds, bm, "dl", reg)

    _g4.DivisionVolume(
        "dv", dl, ml, _g4.DivisionVolume.Axis.kXAxis, 5, 20, 0, reg, True, "mm"
    )

    assert float(ms.pX) == 100.0
    assert float(ms.pY) == 100.0
    assert float(ms.pZ) == 100.0


def test_DivisionMeshesDoNotMutateMotherSolidTubs():
    reg = _g4.Registry()
    mrmin = _gd.Constant("mrmin", "0", reg, True)
    mrmax = _gd.Constant("mrmax", "50", reg, True)
    mdz = _gd.Constant("mdz", "60", reg, True)
    msphi = _gd.Constant("msphi", "0", reg, True)
    mdphi = _gd.Constant("mdphi", "2*pi", reg, True)

    ms = _g4.solid.Tubs("ms", mrmin, mrmax, mdz, msphi, mdphi, reg, "mm", "rad")
    ds = _g4.solid.Tubs("ds", 0, 10, 60, 0, 2 * _np.pi, reg, "mm", "rad")

    wm = _g4.MaterialPredefined("G4_Galactic")
    bm = _g4.MaterialPredefined("G4_Fe")
    ml = _g4.LogicalVolume(ms, wm, "ml", reg)
    dl = _g4.LogicalVolume(ds, bm, "dl", reg)

    _g4.DivisionVolume(
        "dv", dl, ml, _g4.DivisionVolume.Axis.kRho, 5, 10, 0, reg, True, "mm"
    )

    assert float(ms.pRMin) == 0.0
    assert float(ms.pRMax) == 50.0
    assert float(ms.pDz) == 60.0
    assert float(ms.pSPhi) == 0.0
    assert float(ms.pDPhi) == 2 * _np.pi


def test_DivisionWithPlainFloatMotherParameters():
    """Solids constructed programmatically carry plain floats, not
    expression-backed parameters - division mesh creation must work for
    them too (previously raised
    AttributeError: 'float' object has no attribute 'expression')."""
    reg = _g4.Registry()

    ms = _g4.solid.Box("ms", 100, 100, 100, reg, "mm")
    ds = _g4.solid.Box("ds", 20, 100, 100, reg, "mm")

    wm = _g4.MaterialPredefined("G4_Galactic")
    bm = _g4.MaterialPredefined("G4_Fe")
    ml = _g4.LogicalVolume(ms, wm, "ml", reg)
    dl = _g4.LogicalVolume(ds, bm, "dl", reg)

    dv = _g4.DivisionVolume(
        "dv", dl, ml, _g4.DivisionVolume.Axis.kXAxis, 5, 20, 0, reg, True, "mm"
    )

    assert len(dv.meshes) == 5
    assert float(dv.meshes[0].solid.pX) == 20.0
    # and the mother is still intact
    assert float(ms.pX) == 100.0
