from urllib.request import urlretrieve
from src.utils.common import h5py, Path

OUT_DIR = Path("../data/")

FILES = {
    "tiny" :{
        "training": OUT_DIR / "galaxiesml_tiny" / "5x64x64_training_reduced_tiny.hdf5",
        "validation": OUT_DIR  / "galaxiesml_tiny" /  "5x64x64_validation_reduced_tiny.hdf5",
        "testing": OUT_DIR  / "galaxiesml_tiny" /  "5x64x64_testing_reduced_tiny.hdf5"
    },
    "small" :{
        "training": OUT_DIR  / "galaxiesml_small" /  "5x64x64_training_reduced_small.hdf5",
        "validation": OUT_DIR  / "galaxiesml_small" /  "5x64x64_validation_reduced_small.hdf5",
        "testing": OUT_DIR  / "galaxiesml_small" /  "5x64x64_testing_reduced_small.hdf5"
    },
    "medium" :{
        "training": OUT_DIR / "galaxiesml_medium" / "5x64x64_training_reduced_medium.hdf5",
        "validation": OUT_DIR / "galaxiesml_medium" / "5x64x64_validation_reduced_medium.hdf5",
        "testing": OUT_DIR / "galaxiesml_medium" / "5x64x64_testing_reduced_medium.hdf5"
    },
    "large" :{
        "training": OUT_DIR / "galaxiesml_large" / "5x64x64_training_reduced_large.hdf5",
        "validation": OUT_DIR / "galaxiesml_large" / "5x64x64_validation_reduced_large.hdf5",
        "testing": OUT_DIR / "galaxiesml_large" / "5x64x64_testing_reduced_large.hdf5"
    }
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


for size, data_paths in FILES.items():
    for split, path in data_paths.items():
        if not path.exists():
            print(f"Missing download for '{split}' in path: '{path}'")
            continue
        structure_file = dump_hdf5_structure(path)
        print(f"Structure written to: {structure_file}")

