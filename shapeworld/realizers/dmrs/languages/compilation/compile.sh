#!/bin/bash

rm -r $2
cp -r $1 $2

python copy_and_modify.py $1 $2

cat lexicon.tdl >> $2/lexicon.tdl
cat abstract.smi >> $2/etc/abstract.smi
cat hierarchy.smi >> $2/etc/hierarchy.smi
cat surface.smi >> $2/etc/surface.smi
cat nogen-rules.set >> $2/lkb/nogen-rules.set
cat nogen-lex.set >> $2/lkb/nogen-lex.set

../../resources/ace -G ../english.dat -g $2/ace/config.tdl
