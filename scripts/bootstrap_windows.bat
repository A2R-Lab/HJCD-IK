@echo off
setlocal

git config url."https://github.com/".insteadOf "git@github.com:"
git config url."https://github.com/".insteadOf "git@github.com/"

git submodule sync --recursive || exit /b 1
git submodule update --init --recursive || exit /b 1

echo [OK] submodules ready
endlocal
