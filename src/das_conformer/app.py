# from .formbuilder import YamlDialog
# from qtpy import QtWidgets, QtCore, QtGui
# import sys
# import rich

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     dialog = YamlDialog(yaml_file="src/das_conformer/das_train.yaml")
#     form_data = dialog.exec()

#     rich.print(form_data)


from magicgui import magicgui
from pathlib import Path

@magicgui(data_path=dict(widget_type="FileEdit", mode="d"))
def train(data_path: Path, spec_hop_time: float):
    """_summary_

    Args:
        data_path (Path): _description_
        spec_hop_time (float): _description_

    Returns:
        _type_: _description_
    """

    print(data_path, spec_hop_time)
    return {"data_path": data_path, "spec_hop_time": spec_hop_time}


params = train.show(run=True)
print(params)
