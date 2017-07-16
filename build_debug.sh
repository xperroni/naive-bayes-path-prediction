#!/bin/bash

mkdir -p build/debug
cd build/debug

cmake -DCMAKE_BUILD_TYPE=Debug ../.. && make

cd -

ln -sf build/debug/nb-predict nb-predict
