
echo "classification:"
echo "  cnn"
python evaluate.py -t classification -n oneshape -m cnn -b 1 -e 2 -i 1 -v 0
echo "  resnet"
python evaluate.py -t classification -n oneshape -m resnet -b 1 -e 2 -i 1 -v 0
echo "  fracnet"
python evaluate.py -t classification -n oneshape -m fracnet -b 1 -e 2 -i 1 -v 0

echo "agreement:"
echo "  cnn_only"
python evaluate.py -t agreement -n oneshape -m cnn_only -b 1 -e 2 -i 1 -v 0
echo "  lstm_only"
python evaluate.py -t agreement -n oneshape -m lstm_only -b 1 -e 2 -i 1 -v 0
echo "  cnn_bow"
python evaluate.py -t agreement -n oneshape -m cnn_bow -b 1 -e 2 -i 1 -v 0
echo "  cnn_lstm"
python evaluate.py -t agreement -n oneshape -m cnn_lstm -b 1 -e 2 -i 1 -v 0
echo "  cnn_conv"
python evaluate.py -t agreement -n oneshape -m cnn_conv -b 1 -e 2 -i 1 -v 0
echo "  resnet_lstm"
python evaluate.py -t agreement -n oneshape -m resnet_lstm -b 1 -e 2 -i 1 -v 0
echo "  fracnet_lstm"
python evaluate.py -t agreement -n oneshape -m fracnet_lstm -b 1 -e 2 -i 1 -v 0

