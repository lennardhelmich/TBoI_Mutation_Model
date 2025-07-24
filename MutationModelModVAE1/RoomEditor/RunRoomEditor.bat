@echo off
title Isaac AI Room Editor v2.3
echo ==========================================
echo    🏠 Isaac AI Room Editor v2.3
echo ==========================================
echo.
echo 🚀 Starting AI-powered room editor...
echo 📊 Final clean build with all improvements
echo 🤖 VAE model generates unique room variations
echo 📁 Using mutation data from Data/Mutations_Extracted/
echo.
echo 💡 The editor will monitor Isaac save files and
echo    generate AI variations when you change rooms!
echo.
echo ⚠️  Make sure Isaac is running and you're in-game
echo.
echo Make sure your save file path is correct:
echo ../../data/mutationmodel/save1.dat
echo.
echo Press Ctrl+C to stop the program
echo.
pause
echo.
".\dist\RoomEditor\RoomEditor.exe"
echo.
echo ✅ RoomEditor has finished executing
echo.
pause
