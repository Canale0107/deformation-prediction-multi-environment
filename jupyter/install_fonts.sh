#!/bin/bash

# フォントをダウンロードしてインストール
wget https://downloads.sourceforge.net/corefonts/andale32.exe
wget https://downloads.sourceforge.net/corefonts/arial32.exe
wget https://downloads.sourceforge.net/corefonts/arialb32.exe
wget https://downloads.sourceforge.net/corefonts/comic32.exe
wget https://downloads.sourceforge.net/corefonts/courie32.exe
wget https://downloads.sourceforge.net/corefonts/georgi32.exe
wget https://downloads.sourceforge.net/corefonts/impact32.exe
wget https://downloads.sourceforge.net/corefonts/times32.exe
wget https://downloads.sourceforge.net/corefonts/trebuc32.exe
wget https://downloads.sourceforge.net/corefonts/verdan32.exe
wget https://downloads.sourceforge.net/corefonts/webdin32.exe

cabextract -q -d /usr/share/fonts/truetype/msttcorefonts andale32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts arial32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts arialb32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts comic32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts courie32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts georgi32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts impact32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts times32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts trebuc32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts verdan32.exe
cabextract -q -d /usr/share/fonts/truetype/msttcorefonts webdin32.exe

fc-cache -fv
rm -f *.exe