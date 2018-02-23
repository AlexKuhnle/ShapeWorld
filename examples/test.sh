echo "agreement:"

echo "  oneshape"
python generate.py -d examples/test -U -t agreement -n oneshape -i 1 -p 0.1 -M -H
echo "  multishape"
python generate.py -d examples/test -U -t agreement -n multishape -i 1 -p 0.1 -M -H
echo "  spatial"
python generate.py -d examples/test -U -t agreement -n spatial -i 1 -p 0.1 -M -H
echo "  relational"
python generate.py -d examples/test -U -t agreement -n relational -i 1 -p 0.1 -M -H
echo "  quantification_simple"
python generate.py -d examples/test -U -t agreement -n quantification_simple -i 1 -p 0.1 -M -H
echo "  quantification"
python generate.py -d examples/test -U -t agreement -n quantification -i 1 -p 0.1 -M -H
echo "  quantification_complex"
python generate.py -d examples/test -U -t agreement -n quantification_complex -i 1 -p 0.1 -M -H
echo "  combination"
python generate.py -d examples/test -U -t agreement -n combination -i 1 -p 0.1 -M -H

echo "  oneshape (python2)"
python2 generate.py -d examples/test -U -t agreement -n oneshape -i 1 -p 0.1 -M -H


echo "classification:"

echo "  oneshape"
python generate.py -d examples/test -U -t classification -n oneshape -i 1 -p 0.1 -M -H
echo "  multishape"
python generate.py -d examples/test -U -t classification -n multishape -i 1 -p 0.1 -M -H
echo "  countshape"
python generate.py -d examples/test -U -t classification -n countshape -i 1 -p 0.1 -M -H

echo "  oneshape (python2)"
python2 generate.py -d examples/test -U -t classification -n oneshape -i 1 -p 0.1 -M -H


rm -r examples/test
