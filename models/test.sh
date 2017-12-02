echo "agreement:"

echo "  custom"
python train.py -t agreement -n oneshape_simple -m custom --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m custom --model-dir models/test/ -b 1 -i 1 -v 0
echo "  cnn_only"
python train.py -t agreement -n oneshape_simple -m cnn_only --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m cnn_only --model-dir models/test/ -b 1 -i 1 -v 0
echo "  lstm_only"
python train.py -t agreement -n oneshape_simple -m lstm_only --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m lstm_only --model-dir models/test/ -b 1 -i 1 -v 0
echo "  cnn_bow"
python train.py -t agreement -n oneshape_simple -m cnn_bow --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m cnn_bow --model-dir models/test/ -b 1 -i 1 -v 0
echo "  cnn_lstm"
python train.py -t agreement -n oneshape_simple -m cnn_lstm --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m cnn_lstm --model-dir models/test/ -b 1 -i 1 -v 0
echo "  cnn_conv"
python train.py -t agreement -n oneshape_simple -m cnn_conv --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m cnn_conv --model-dir models/test/ -b 1 -i 1 -v 0
echo "  resnet_lstm"
python train.py -t agreement -n oneshape_simple -m resnet_lstm --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m resnet_lstm --model-dir models/test/ -b 1 -i 1 -v 0
echo "  fracnet_lstm"
python train.py -t agreement -n oneshape_simple -m fracnet_lstm --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m fracnet_lstm --model-dir models/test/ -b 1 -i 1 -v 0

echo "  cnn_only (python2)"
python2 train.py -t agreement -n oneshape_simple -m cnn_only --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t agreement -n oneshape_simple -m cnn_only --model-dir models/test/ -b 1 -i 1 -v 0



echo "classification:"

echo "  cnn"
python train.py -t classification -n oneshape -m cnn --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t classification -n oneshape -m cnn --model-dir models/test/ -b 1 -i 1 -v 0
echo "  resnet"
python train.py -t classification -n oneshape -m resnet --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t classification -n oneshape -m resnet --model-dir models/test/ -b 1 -i 1 -v 0
echo "  fracnet"
python train.py -t classification -n oneshape -m fracnet --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t classification -n oneshape -m fracnet --model-dir models/test/ -b 1 -i 1 -v 0

echo "  cnn (python2)"
python train.py -t classification -n oneshape -m cnn --model-dir models/test/ -b 1 -e 2 -i 1 -v 0
python evaluate.py -t classification -n oneshape -m cnn --model-dir models/test/ -b 1 -i 1 -v 0



rm -r models/test
