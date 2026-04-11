from urllib.request import urlretrieve
from utils.common import h5py, Path

OUT_DIR = Path("../data/galaxiesml")

FILES = {
    "training": OUT_DIR / "5x64x64_training_with_morphology.hdf5",
    "validation": OUT_DIR / "5x64x64_validation_with_morphology.hdf5",
    "testing": OUT_DIR / "5x64x64_testing_with_morphology.hdf5"
}

def dump_hdf5_structure(path: Path) -> Path:
    out_txt = path.with_suffix(path.suffix + ".structure.txt")

    def write_attrs(obj, fh, indent):
        if len(obj.attrs) == 0:
            return
        fh.write(f"{indent}@attrs\n")
        for k, v in obj.attrs.items():
            fh.write(f"{indent}  - {k}: {repr(v)}\n")

    with h5py.File(path, "r") as f, open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"FILE: {path}\n\n")

        def recurse(name, obj, depth=0):
            indent = "  " * depth

            if isinstance(obj, h5py.Group):
                fh.write(f"{indent}[GROUP] {name or '/'}\n")
                write_attrs(obj, fh, indent + "  ")

                for key in obj.keys():
                    child = obj[key]
                    child_name = f"{name}/{key}" if name else key
                    recurse(child_name, child, depth + 1)

            elif isinstance(obj, h5py.Dataset):
                fh.write(
                    f"{indent}[DATASET] {name} | "
                    f"shape={obj.shape} | dtype={obj.dtype}\n"
                )
                write_attrs(obj, fh, indent + "  ")

        recurse("", f, 0)

    return out_txt


for name, out_path in FILES.items():
    if not out_path.resolve():
        print(f"Missing download for '{name}' in path: '{out_path}'")
    structure_file = dump_hdf5_structure(out_path)
    print(f"Structure written to: {structure_file}")