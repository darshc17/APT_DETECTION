from cx_Freeze import setup, Executable

# List additional files and folders to include
include_files = [
    ("templates", "templates"),
    ("static", "static"),
    ("models", "models"),
    ("flow", "flow"),
    ("input_logs.csv", "input_logs.csv"),
    ("output_logs.csv", "output_logs.csv"),
]

build_exe_options = {
    "packages": ["os", "flask"],
    "include_files": include_files,
    "include_msvcr": True  # helps avoid DLL errors on Windows
}

setup(
    name="APT Detection System",
    version="1.0",
    description="Advanced Persistent Threat Detection System",
    options={"build_exe": build_exe_options},
    executables=[Executable("application.py", base=None)]
)
