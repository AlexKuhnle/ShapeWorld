# agreement
rm -r examples/agreement
python generate.py -d examples/agreement/oneshape_simple -U -t agreement -n oneshape_simple -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/oneshape -U -t agreement -n oneshape -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/multishape_simple -U -t agreement -n multishape_simple -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/multishape -U -t agreement -n multishape -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/spatial_simple -U -t agreement -n spatial_simple -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/spatial -U -t agreement -n spatial -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/relational -U -t agreement -n relational -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/maxattr -U -t agreement -n maxattr -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/quantification_count_simple -U -t agreement -n quantification_count_simple -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/quantification_count -U -t agreement -n quantification_count -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/quantification_ratio_simple -U -t agreement -n quantification_ratio_simple -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/quantification_ratio -U -t agreement -n quantification_ratio -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/quantification_simple -U -t agreement -n quantification_simple -i 100 -p 0.1 -M -H
python generate.py -d examples/agreement/combination -U -t agreement -n combination -i 100 -p 0.1 -M -H

# classification
rm -r examples/classification
python generate.py -d examples/classification/oneshape -U -t classification -n oneshape -i 100 -p 0.1 -M -H
python generate.py -d examples/classification/multishape -U -t classification -n multishape -i 100 -p 0.1 -M -H
python generate.py -d examples/classification/countshape -U -t classification -n countshape -i 100 -p 0.1 -M -H

# readme data
rm -r examples/readme
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n multishape -f "(5,5,1,1)" -i 100 -M
python generate.py -d examples/test1 -a tar:bzip2 -t agreement -n multishape -c "load(examples/readme)" -f "(2,2,1,1)" -i 100 -M
python generate.py -d examples/test2 -a tar:bzip2 -t agreement -n multishape -c "load(examples/test1)" -f "(1,1,1,1)" -i 100 -M
rm -r examples/test1
rm -r examples/test2

# load and evaluate
python train.py -t agreement -n multishape -c "load(examples/readme)" -m cnn_bow -i 10 -T
