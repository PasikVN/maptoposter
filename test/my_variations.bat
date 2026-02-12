@echo off

chcp 65001

cd %~dp0
cd ..

py -m pip install -r requirements.txt

::timeout /t 2
::cls

::py create_map_poster.py -c "Hoi An" -dc "Hội An" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 15000 -W 16 -H 12 -lat 15.87923 -long 108.35067
::py create_map_poster.py -c "Hoi An" -dc "Hội An" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 15000 -W 16 -H 12 -lat 15.87923 -long 108.35067 --include-oceans

::py create_map_poster.py -c "Hue" -dc "Huế" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 15000 -W 16 -H 12
::py create_map_poster.py -c "Hue" -dc "Huế" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 15000 -W 16 -H 12
::
::py create_map_poster.py -c "Hanoi" -dc "Hà Nội" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 35000 -W 16 -H 12
::py create_map_poster.py -c "Hanoi" -dc "Hà Nội" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 35000 -W 16 -H 12
::
::py create_map_poster.py -c "Ho Chi Minh city" -dc "Thành phố Hồ Chí Minh" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 25000 -W 16 -H 12
::py create_map_poster.py -c "Ho Chi Minh city" -dc "Thành phố Hồ Chí Minh" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 25000 -W 16 -H 12
::
::py create_map_poster.py -c "van lam" -dc "Hoa Lư" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 4000 -d 10000 -W 16 -H 12
::py create_map_poster.py -c "van lam" -dc "Hoa Lư" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 4000 -d 10000 -W 16 -H 12
::
::py create_map_poster.py -c "Ha Long" -dc "Hạ Long Bay" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 25000 -W 16 -H 12 -lat 20.90725 -long 107.14077
::py create_map_poster.py -c "Ha Long" -dc "Hạ Long Bay" -C "Vietnam" -dC "Việt Nam" -t japanese_ink -d 25000 -W 16 -H 12 -lat 20.90725 -long 107.14077


:: =======================  DEBUG ============================
py create_map_poster.py -c "Tokyo" -C "Japan" -dc "東京" -dC "日本" --font-family "Noto Sans JP" -t japanese_ink -d 40000 -iR
::py create_map_poster.py -c "Grenoble" -C "France" -t japanese_ink -d 18000 --fast
::py create_map_poster.py -c "London" -C "England" -t pascal -d 18000
::py create_map_poster.py -c "Los Angeles" -C "United States of America" -t gta -d 18000 -iR
::py create_map_poster.py -c "La Rochelle" -C "France" -t mohe_subaraya_light -d 18000 --include-oceans --include-railways
::py create_map_poster.py -c "Brest" -C "France" -t mohe_subaraya_light -d 18000 -iO -iR
py create_map_poster.py -c "Ho Chi Minh city" -dc "Thành phố Hồ Chí Minh" -C "Vietnam" -dC "Việt Nam" -t mohe_subaraya_light -d 40000 -iO -iR --font-family "Playfair Display"

::py create_map_poster.py -c "SPA-Francorchamps" -C "Belgium" -t grand_prix -d 6000 --show-north -O -90 --font-family "Russo One" --hide-north -f svg
py create_map_poster.py -c "Warszawa" -C "Polska" -t grand_prix -d 60000 --font-family "Brygada 1918"


pause