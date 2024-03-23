for /d /r . %%d in (__pycache__) do @if exist "%%d" echo "%%d" && rd /s /q "%%d"
for /d /r . %%d in (.vs) do @if exist "%%d" echo "%%d" && rd /s /q "%%d"
for /d /r . %%d in (bin) do @if exist "%%d" echo "%%d" && rd /s /q "%%d"
for /d /r . %%d in (build) do @if exist "%%d" echo "%%d" && rd /s /q "%%d"
pause