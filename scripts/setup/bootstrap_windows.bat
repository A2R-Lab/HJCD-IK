@echo off
setlocal EnableExtensions

REM --- scripts/setup is two directories below the repository root. ---
cd /d "%~dp0\..\.." || exit /b 1

echo [bootstrap] initialize top-level build and Panda collision dependencies...
git submodule update --init external/GLASS external/GRiD external/foam || exit /b 1

REM Use HTTPS for nested codegen dependencies without editing tracked .gitmodules.
REM RBDReference is not needed for building or codegen. Preserve the committed pins.
echo [bootstrap] initialize GRiD codegen dependencies...
git -C external/GRiD -c submodule.GLASS.url=https://github.com/A2R-Lab/GLASS.git -c submodule.URDFParser.url=https://github.com/A2R-Lab/URDFParser.git submodule update --init external/GLASS external/URDFParser || exit /b 1

echo [OK] submodules ready
endlocal
