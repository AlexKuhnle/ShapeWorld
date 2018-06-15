echo "agreement:"

echo "  custom"
python -O train.py -t agreement -n existential -m custom -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m custom -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  always_true"
python -O train.py -t agreement -n existential -m always_true -b 1 -i 2 -e 2 -f 2 --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m always_true -b 2 -i 1 --report-file models/test.csv --verbosity 0
echo "  always_false"
python -O train.py -t agreement -n existential -m always_false -b 1 -i 2 -e 2 -f 2 --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m always_false -b 2 -i 1 --report-file models/test.csv --verbosity 0
echo "  prefix_prior"
python -O train.py -t agreement -n existential -m prefix_prior -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m prefix_prior -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  suffix_prior"
python -O train.py -t agreement -n existential -m suffix_prior -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m suffix_prior -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  type_prior"
python -O train.py -t agreement -n existential -m type_prior -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m type_prior -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_only"
python -O train.py -t agreement -n existential -m cnn_only -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_only -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  rnn_only"
python -O train.py -t agreement -n existential -m rnn_only -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m rnn_only -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_bow"
python -O train.py -t agreement -n existential -m cnn_bow -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_bow -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_rnn"
python -O train.py -t agreement -n existential -m cnn_rnn -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_rnn -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  resnet_rnn"
python -O train.py -t agreement -n existential -m resnet_rnn -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m resnet_rnn -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
# echo "  fracnet_rnn"
# python -O train.py -t agreement -n existential -m fracnet_rnn -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
# python -O evaluate.py -t agreement -n existential -m fracnet_rnn -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_conv"
python -O train.py -t agreement -n existential -m cnn_conv -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_rnn_att1"
python -O train.py -t agreement -n existential -m cnn_rnn_att1 -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_rnn_att1 -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_rnn_att2"
python -O train.py -t agreement -n existential -m cnn_rnn_att2 -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_rnn_att2 -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_rnn_san"
python -O train.py -t agreement -n existential -m cnn_rnn_san -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_rnn_san -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_conv_san"
python -O train.py -t agreement -n existential -m cnn_conv_san -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_conv_san -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  cnn_rnn_film"
python -O train.py -t agreement -n existential -m cnn_rnn_film -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t agreement -n existential -m cnn_rnn_film -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
# echo "  cnn_rnn_rel"
#python -O train.py -t agreement -n existential -m cnn_rnn_rel -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
#python -O evaluate.py -t agreement -n existential -m cnn_rnn_rel -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0

echo "  cnn_only (python2)"
python2 -O train.py -t agreement -n existential -m cnn_only -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python2 -O evaluate.py -t agreement -n existential -m cnn_only -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0



echo "classification:"

echo "  cnn"
python -O train.py -t classification -n shape -m cnn -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t classification -n shape -m cnn -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  resnet"
python -O train.py -t classification -n shape -m resnet -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t classification -n shape -m resnet -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0
echo "  fracnet"
python -O train.py -t classification -n shape -m fracnet -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python -O evaluate.py -t classification -n shape -m fracnet -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0

echo "  cnn (python2)"
python2 -O train.py -t classification -n shape -m cnn -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv --verbosity 0 -Y
python2 -O evaluate.py -t classification -n shape -m cnn -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv --verbosity 0



echo "readme:"
python -O train.py -t agreement -n existential -v readme -c examples/readme -m cnn_bow -i 10 -T --model-dir models/test/ -Y
python -O evaluate.py -t agreement -n existential -v readme -c examples/readme -m cnn_bow -i 10 --model-dir models/test/



rm -r models/test
rm -r models/summary
rm  models/test.csv
