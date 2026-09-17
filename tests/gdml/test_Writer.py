import xml.dom.minidom as _minidom

import pyg4ometry.gdml as _gd
import pyg4ometry.geant4 as _g4


def test_DivisionVolumeOffsetWrittenCorrectly(tmp_path):
    """The divisionvol offset attribute must hold the division offset,
    not the division width (regression test for a copy-paste bug where
    offset was written as width)."""
    reg = _g4.Registry()

    # mother solid via Constants (DivisionVolume mesh creation needs
    # expression-backed parameters)
    mbx = _gd.Constant("mbx", "100", reg, True)
    mby = _gd.Constant("mby", "100", reg, True)
    mbz = _gd.Constant("mbz", "100", reg, True)

    wm = _g4.MaterialPredefined("G4_Galactic")
    bm = _g4.MaterialPredefined("G4_Fe")

    ws = _g4.solid.Box("ws", 1000, 1000, 1000, reg, "mm")
    ms = _g4.solid.Box("ms", mbx, mby, mbz, reg, "mm")
    ds = _g4.solid.Box("ds", 100, 100, 10, reg, "mm")

    wl = _g4.LogicalVolume(ws, wm, "wl", reg)
    ml = _g4.LogicalVolume(ms, wm, "ml", reg)
    dl = _g4.LogicalVolume(ds, bm, "dl", reg)

    _g4.DivisionVolume(
        "dv", dl, ml, _g4.DivisionVolume.Axis.kZAxis, 8, 10, 5, reg, True, "mm"
    )
    _g4.PhysicalVolume([0, 0, 0], [0, 0, 0], ml, "ml_pv", wl, reg)
    reg.setWorld(wl.name)

    out = tmp_path / "division_offset.gdml"
    w = _gd.Writer()
    w.addDetector(reg)
    w.write(out)

    doc = _minidom.parse(str(out))
    dvols = doc.getElementsByTagName("divisionvol")
    assert len(dvols) == 1
    assert float(dvols[0].getAttribute("width")) == 10.0
    assert float(dvols[0].getAttribute("offset")) == 5.0
