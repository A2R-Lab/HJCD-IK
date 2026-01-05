@echo off
setlocal EnableExtensions

REM --- Go to repo root (directory of this script's parent) ---
cd /d "%~dp0\.."

echo [bootstrap] syncing top-level submodules...
git submodule sync --recursive || exit /b 1

echo [bootstrap] init/update GRiD submodule (top-level)...
git submodule update --init external/GRiD || exit /b 1

echo [bootstrap] rewriting GRiD nested submodule URLs to HTTPS...
git config -f external/GRiD/.gitmodules submodule.GRiDCodeGenerator.url https://github.com/A2R-Lab/GRiDCodeGenerator.git || exit /b 1
git config -f external/GRiD/.gitmodules submodule.RBDReference.url      https://github.com/A2R-Lab/RBDReference.git      || exit /b 1
git config -f external/GRiD/.gitmodules submodule.URDFParser.url        https://github.com/A2R-Lab/URDFParser.git        || exit /b 1

echo [bootstrap] syncing GRiD nested submodules...
git -C external/GRiD submodule sync --recursive || exit /b 1

echo [bootstrap] init/update all nested submodules recursively...
git submodule update --init --recursive || exit /b 1

echo [OK] submodules ready
endlocal
