@echo off

REM 在批命令中使用 @echo off 可以关闭打印命令本身的信息，使命令执行更加简洁

for /d %%i in (*) do (

  REM 使用 for 命令遍历当前文件夹下所有子文件夹

  REM /d 参数表示只处理文件夹，而不处理文件

  cd "%%i"

  REM 切换到当前子文件夹内

  if exist bin (

    REM 判断当前子文件夹内是否存在bin目录

    echo Deleting bin directory in %%i...

    REM 打印当前正在执行的操作提示信息

    rd /s /q bin

    REM 使用 rd 命令删除 bin 目录

  )
  

  if exist build (

    REM 判断当前子文件夹内是否存在bin目录

    echo Deleting bin directory build %%i...

    REM 打印当前正在执行的操作提示信息

    rd /s /q build

    REM 使用 rd 命令删除 bin 目录

  )

  if exist third (

    REM 判断当前子文件夹内是否存在bin目录

    echo Deleting bin directory third %%i...

    REM 打印当前正在执行的操作提示信息

    rd /s /q third

    REM 使用 rd 命令删除 bin 目录

  )

  if exist files (

    REM 判断当前子文件夹内是否存在bin目录

    echo Deleting bin directory files %%i...

    REM 打印当前正在执行的操作提示信息

    rd /s /q files

    REM 使用 rd 命令删除 bin 目录

  )

  if exist .vs (

    REM 判断当前子文件夹内是否存在bin目录

    echo Deleting bin directory .vs %%i...

    REM 打印当前正在执行的操作提示信息

    rd /s /q .vs

    REM 使用 rd 命令删除 bin 目录

  )

  if exist x64 (

    REM 判断当前子文件夹内是否存在bin目录

    echo Deleting bin directory x64 %%i...

    REM 打印当前正在执行的操作提示信息

    rd /s /q x64

    REM 使用 rd 命令删除 bin 目录

  )

  cd ..

  REM 切换回上级目录

)

echo All bin directories have been deleted.

REM 打印操作完成提示信息

pause

REM 暂停脚本执行，以便查看执行结果