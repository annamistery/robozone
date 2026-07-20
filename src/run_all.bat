@echo off
chcp 65001 >nul
python -m src.extract_frames || exit /b 1
python -m src.autolabel --preview || exit /b 1
echo.
echo === Проверь глазами work\preview, потом нажми любую клавишу ===
pause
python -m src.build_dataset || exit /b 1
python -m src.train || exit /b 1
echo Готово. Веса в work\runs\conveyor_seg_v1\weights\best.pt
