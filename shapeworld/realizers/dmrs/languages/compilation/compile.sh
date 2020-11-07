#!/bin/bash

rm -r $2

mkdir $2
cp $1/auxverbs.tdl $2
cp $1/constructions.tdl $2
cp $1/ctype.tdl $2
cp $1/delims.tdl $2
cp $1/dts.tdl $2
cp $1/english.tdl $2/
cp $1/fundamentals.tdl $2
cp $1/gle.tdl $2
cp $1/idioms.mtr $2
cp $1/inflr.tdl $2
cp $1/inflr-pnct.tdl $2
cp $1/irregs.tab $2
cp $1/letypes.tdl $2
# cp $1/lexicon.tdl $2
cp $1/lexrinst.tdl $2
cp $1/lexrules.tdl $2
cp $1/lextypes.tdl $2
cp $1/lfr.tdl $2
# cp $1/mtr.tdl $2/
cp $1/parse-nodes.tdl $2
cp $1/redwoods.mem $2
cp $1/roots.tdl $2
cp $1/semi.vpm $2
cp $1/syntax.tdl $2
cp $1/tmt.tdl $2
# cp $1/trigger.mtr $2
cp $1/Version.lsp $2/

mkdir $2/ace
cp $1/ace/ace-erg-qc.txt $2/ace/
cp $1/ace/config.tdl $2/ace/
cp $1/ace/english-postagger.hmm $2/ace/

mkdir $2/etc
cp $1/etc/abstract.smi $2/etc/
cp $1/etc/erg.smi $2/etc/
cp $1/etc/hierarchy.smi $2/etc/
cp $1/etc/surface.smi $2/etc/

mkdir $2/lkb
cp $1/lkb/nogen-lex.set $2/lkb/
cp $1/lkb/nogen-rules.set $2/lkb/

mkdir $2/rpp
cp $1/rpp/ascii.rpp $2/rpp/
cp $1/rpp/gml.rpp $2/rpp/
cp $1/rpp/html.rpp $2/rpp/
cp $1/rpp/quotes.rpp $2/rpp/
cp $1/rpp/tokenizer.rpp $2/rpp/
cp $1/rpp/wiki.rpp $2/rpp/
cp $1/rpp/xml.rpp $2/rpp/

mkdir $2/tmr
cp $1/tmr/class.tdl $2/tmr/
cp $1/tmr/finis.tdl $2/tmr/
cp $1/tmr/gml.tdl $2/tmr/
cp $1/tmr/ne1.tdl $2/tmr/
cp $1/tmr/ne2.tdl $2/tmr/
cp $1/tmr/ne3.tdl $2/tmr/
cp $1/tmr/pos.tdl $2/tmr/
cp $1/tmr/post-generation.tdl $2/tmr/
cp $1/tmr/ptb.tdl $2/tmr/
cp $1/tmr/punctuation.tdl $2/tmr/
cp $1/tmr/spelling.tdl $2/tmr/
cp $1/tmr/split.tdl $2/tmr/

python copy_and_modify.py $1 $2

cat lexicon.tdl >> $2/lexicon.tdl
cat abstract.smi >> $2/etc/abstract.smi
cat hierarchy.smi >> $2/etc/hierarchy.smi
cat surface.smi >> $2/etc/surface.smi
cat nogen-rules.set >> $2/lkb/nogen-rules.set
cat nogen-lex.set >> $2/lkb/nogen-lex.set

../../resources/ace -G ../english.dat -g $2/ace/config.tdl
