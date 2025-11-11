@echo off
chcp 65001 >nul
title Stencil Analyzer - DEBUG MODE

echo ========================================
echo    Stencil Analyzer - DEBUG MODE
echo ========================================
echo.

echo Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    pause
    exit /b 1
)

echo Запуск в режиме отладки...
echo.

python main.py --debug

if errorlevel 1 (
    echo.
    echo ❌ Ошибка выполнения (код: %errorlevel%)
) else (
    echo.
    echo ✅ Отладка завершена
)

pause