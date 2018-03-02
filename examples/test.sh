echo "agreement:"

echo "  oneshape"
python generate.py -d examples/test1 -U -t agreement -n oneshape -i 1 -M -H
echo "  multishape"
python generate.py -d examples/test1 -U -t agreement -n multishape -i 1 -M -H
echo "  spatial"
python generate.py -d examples/test1 -U -t agreement -n spatial -i 1 -M -H
echo "  relational"
python generate.py -d examples/test1 -U -t agreement -n relational -i 1 -M -H
echo "  quantification_simple"
python generate.py -d examples/test1 -U -t agreement -n quantification_simple -i 1 -M -H
echo "  quantification"
python generate.py -d examples/test1 -U -t agreement -n quantification -i 1 -M -H
echo "  quantification_complex"
python generate.py -d examples/test1 -U -t agreement -n quantification_complex -i 1 -M -H
echo "  combination"
python generate.py -d examples/test1 -U -t agreement -n combination -i 1 -M -H

echo "  oneshape (python2)"
python2 generate.py -d examples/test1 -U -t agreement -n oneshape -i 1 -M -H


echo "classification:"

echo "  oneshape"
python generate.py -d examples/test1 -U -t classification -n oneshape -i 1 -M -H
echo "  multishape"
python generate.py -d examples/test1 -U -t classification -n multishape -i 1 -M -H
echo "  countshape"
python generate.py -d examples/test1 -U -t classification -n countshape -i 1 -M -H

echo "  oneshape (python2)"
python2 generate.py -d examples/test1 -U -t classification -n oneshape -i 1 -M -H


echo "generation modes:"

echo "  generate from loaded"
python generate.py -d examples/test1 -t agreement -n multishape -c examples/readme -s 2,1,3 -i 100 -M -T
python generate.py -d examples/test2 -t agreement -n multishape -c examples/test1 -s 1,2,3 -i 100 -M
echo "  generate mixer"
python generate.py -d examples/test1 -t agreement -n multishape,spatial -s 3,2,1 -i 100 -M -Y
echo "  generate mixer from loaded"
python generate.py -d examples/test2 -t agreement -n multishape,spatial -c examples/readme -s 3,2,1 -i 100 -M -Y
echo "  generate from config"
python generate.py -d examples/test2 -c examples/test1/agreement-multishape+spatial.json -s 1,2,3 -i 100 -M -Y
echo "  generate mixer from configs"
python generate.py -d examples/test2 -c examples/test1/agreement-multishape.json,examples/test1/agreement-multishape+spatial.json -s 1,2,3 -i 100 -M -T -Y
echo "  train from previously generated data"
python train.py -t agreement -n multishape -c examples/readme -m cnn_bow -b 2 -i 2 -e 2 -f 2
python train.py -t agreement -n spatial -c examples/readme -m cnn_bow -b 2 -i 2 -e 2 -f 2
python train.py -t agreement -n multishape+spatial -c examples/test1 -m cnn_bow -b 2 -i 2 -e 2 -f 2
python train.py -t agreement -n multishape+multishape+spatial -c examples/test2 -m cnn_bow -b 2 -i 2 -e 2 -f 2 -T


rm -r examples/test1
rm -r examples/test2
