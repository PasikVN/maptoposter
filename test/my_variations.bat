@echo off

chcp 65001

cd %~dp0
cd ..

::py -m pip install -r requirements.txt

::timeout /t 2
cls

::py create_map_poster.py -c "Hoi An" -dc "Hội An" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 15000 -W 16 -H 12 -lat 15.87923 -long 108.35067
::py create_map_poster.py -c "Hoi An" -dc "Hội An" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 15000 -W 16 -H 12 -lat 15.87923 -long 108.35067 --include-oceans

::py create_map_poster.py -c "Hue" -dc "Huế" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 15000 -W 16 -H 12 -iR
::py create_map_poster.py -c "Hue" -dc "Huế" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 15000 -W 16 -H 12

::py create_map_poster.py -c "Hanoi" -dc "Hà Nội" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 35000 -W 16 -H 12 -iR
::py create_map_poster.py -c "Hanoi" -dc "Hà Nội" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 35000 -W 16 -H 12

::py create_map_poster.py -c "van lam" -dc "Hoa Lư" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 4000 -d 10000 -W 16 -H 12
::py create_map_poster.py -c "van lam" -dc "Hoa Lư" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 4000 -d 10000 -W 16 -H 12

::py create_map_poster.py -c "Ha Long" -dc "Hạ Long Bay" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 25000 -W 16 -H 12 -lat 20.90725 -long 107.14077
::py create_map_poster.py -c "Ha Long" -dc "Hạ Long Bay" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 25000 -W 16 -H 12 -lat 20.90725 -long 107.14077


:: =======================  TESTS ============================
@REM :: Testing special font, UTF-8 and city/country forced naming with railways
@REM py create_map_poster.py -c "Tokyo" -C "Japan" -dc "東京" -dC "日本" --font-family "Noto Sans JP" -t japanese_ink -d 40000 -iR
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

@REM :: testing ROAD/RAILWAYS/AEROWAYS (this will test caching): -iR should be defined first because when cached it will not load railways when fetch for the 2nd time)
@REM py create_map_poster.py -c "Changi Airport" -C "Singapore" -t pastel_dream -d 30000 --font-family "Montserrat" -iR
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

@REM :: testing ROAD/***NO RAILWAYS***/AEROWAYS with a rotation and badge displayed automaticaly  (this will test caching)
@REM py create_map_poster.py -c "Changi Airport" -C "Singapore" -t pastel_dream -d 30000 --font-family "Montserrat" -O 45
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

@REM :: testing ROAD/RAILWAYS/AEROWAYS with rotation and no badge  (this will test caching)
@REM py create_map_poster.py -c "Changi Airport" -C "Singapore" -t pastel_dream -d 30000 --font-family "Montserrat" -iR -O 45 --no-show-north
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

@REM :: testing RACEWAYS
@REM py create_map_poster.py -c "Circuit de la Sarthe, Le Mans" -dc "LE MANS" -C "France" -t pastel_dream -d 15000  --font-family "Michroma" -iR
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

@REM :: testing RACEWAYS, LAT/LONG
@REM py create_map_poster.py -c "Nürburgring" -C "Germany" -t grand_prix_dimmed -d 15000 --font-family "Russo One" -lat 50.34765 -long 6.96651
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

@REM :: Testing RACEWAYS, LAT/LONG but with rotation
@REM py create_map_poster.py -c "Nürburgring" -C "Germany" -t grand_prix_dimmed -d 15000 --font-family "Russo One" -lat 50.34765 -long 6.96651 -O 25
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

@REM :: Testing WIDTH/HEIGHT
@REM py create_map_poster.py -c "Ho Chi Minh city" -dc "Thành phố Hồ Chí Minh" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 40000 -iR --font-family "Playfair Display" -W 16 -H 12
@REM if %ERRORLEVEL% NEQ 0 goto :error_handler

:: testing ???
py create_map_poster.py -c "Saint-Martin-de-Ré" -C "France" -t warm_beige -d 5000  --font-family "IM Fell English SC" -iR -iH
if %ERRORLEVEL% NEQ 0 goto :error_handler

echo.
echo ====================================================
echo All posters generated successfully!
echo ====================================================
pause
exit /b 0

:error_handler
echo.
echo ====================================================
echo [!] Stopping: A failure occurred during generation.
echo ====================================================
pause
exit /b %ERRORLEVEL%


