echo "agreement:"

echo "  custom"
python train.py -t agreement -n oneshape -m custom -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0
python evaluate.py -t agreement -n oneshape -m custom -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_only"
python train.py -t agreement -n oneshape -m cnn_only -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_only -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  lstm_only"
python train.py -t agreement -n oneshape -m lstm_only -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m lstm_only -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_bow"
python train.py -t agreement -n oneshape -m cnn_bow -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_bow -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_lstm"
python train.py -t agreement -n oneshape -m cnn_lstm -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_lstm -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  resnet_lstm"
python train.py -t agreement -n oneshape -m resnet_lstm -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m resnet_lstm -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  fracnet_lstm"
python train.py -t agreement -n oneshape -m fracnet_lstm -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m fracnet_lstm -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_conv"
python train.py -t agreement -n oneshape -m cnn_conv -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_lstm_att1"
python train.py -t agreement -n oneshape -m cnn_lstm_att1 -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_lstm_att2"
python train.py -t agreement -n oneshape -m cnn_lstm_att2 -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_lstm_san"
python train.py -t agreement -n oneshape -m cnn_lstm_san -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_conv_san"
python train.py -t agreement -n oneshape -m cnn_conv_san -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_lstm_film"
python train.py -t agreement -n oneshape -m cnn_lstm_film -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  cnn_lstm_rel"
python train.py -t agreement -n oneshape -m cnn_lstm_rel -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0

echo "  cnn_only (python2)"
python2 train.py -t agreement -n oneshape -m cnn_only -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t agreement -n oneshape -m cnn_only -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0



echo "classification:"

echo "  cnn"
python train.py -t classification -n oneshape -m cnn -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t classification -n oneshape -m cnn -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  resnet"
python train.py -t classification -n oneshape -m resnet -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t classification -n oneshape -m resnet -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0
echo "  fracnet"
python train.py -t classification -n oneshape -m fracnet -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t classification -n oneshape -m fracnet -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0

echo "  cnn (python2)"
python train.py -t classification -n oneshape -m cnn -b 1 -i 2 -e 2 -f 2 --model-dir models/test/ --summary-dir models/summary/ --report-file models/test.csv -v 0 -Y
python evaluate.py -t classification -n oneshape -m cnn -b 2 -i 1 --model-dir models/test/ --report-file models/test.csv -v 0


rm -r models/test
rm -r models/summary
rm  models/test.csv
