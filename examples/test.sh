echo "agreement:"

echo "  oneshape_simple"
python generate.py -d examples/test -U -t agreement -n oneshape_simple -i 1 -p 0.1 -M -H
echo "  oneshape"
python generate.py -d examples/test -U -t agreement -n oneshape -i 1 -p 0.1 -M -H
echo "  multishape_simple"
python generate.py -d examples/test -U -t agreement -n multishape_simple -i 1 -p 0.1 -M -H
echo "  multishape"
python generate.py -d examples/test -U -t agreement -n multishape -i 1 -p 0.1 -M -H
echo "  spatial_simple"
python generate.py -d examples/test -U -t agreement -n spatial_simple -i 1 -p 0.1 -M -H
echo "  spatial"
python generate.py -d examples/test -U -t agreement -n spatial -i 1 -p 0.1 -M -H
echo "  relational"
python generate.py -d examples/test -U -t agreement -n relational -i 1 -p 0.1 -M -H
echo "  maxattr"
python generate.py -d examples/test -U -t agreement -n maxattr -i 1 -p 0.1 -M -H
echo "  quantification_count_simple"
python generate.py -d examples/test -U -t agreement -n quantification_count_simple -i 1 -p 0.1 -M -H
echo "  quantification_count"
python generate.py -d examples/test -U -t agreement -n quantification_count -i 1 -p 0.1 -M -H
echo "  quantification_ratio_simple"
python generate.py -d examples/test -U -t agreement -n quantification_ratio_simple -i 1 -p 0.1 -M -H
echo "  quantification_ratio"
python generate.py -d examples/test -U -t agreement -n quantification_ratio -i 1 -p 0.1 -M -H
echo "  quantification_simple"
python generate.py -d examples/test -U -t agreement -n quantification_simple -i 1 -p 0.1 -M -H
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
