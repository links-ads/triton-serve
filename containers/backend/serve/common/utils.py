import zipfile
from pathlib import Path
import io


def unzip_model(model_bytes: str, name: str, version: int):
    

    parent_folder = Path(__file__).parent
    # create a temporary folder to unzip the model
    temp_folder = Path(parent_folder, "temp", name + str(version))
    

    # if temp folder already exists, return None
    if temp_folder.exists():
        return "Request conflict"
        
    zip_data = io.BytesIO(model_bytes)

    try:
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            zip_ref.extractall(temp_folder)
    except:
        return "zip file error"

    all_zip = list(temp_folder.glob("**/*"))
    files_in_zip = [x for x in all_zip if x.is_file()]

    # check if there is minimum 1 or max 2 files
    if len(files_in_zip) < 1 or len(files_in_zip) > 2:
        return "File number error"

    names = []
    
    for f in files_in_zip:
        names.append(f.name)
    
    # check if a file is model.onnx
    if not "model.onnx" in names:
        return "No model.onnx"

    if len(files_in_zip) == 2 and not "config.pbtxt" in names:
        return "No config.pbtxt"
    
    reordered_files = reorder_files(files_in_zip)

    

    # return the path to the model.onnx file and the path to the model.pbtxt file
    return reordered_files


def reorder_files(paths):
    # put the file that ends with .onnx as first element
    if paths[0].name.endswith(".onnx"):
        return paths
    else:
        return [paths[1], paths[0]]