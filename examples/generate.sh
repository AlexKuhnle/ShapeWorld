# agreement
rm -r examples/agreement
python generate.py -d examples/agreement/oneshape -U -t agreement -n oneshape -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/multishape -U -t agreement -n multishape -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/spatial -U -t agreement -n spatial -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/relational -U -t agreement -n relational -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification_simple -U -t agreement -n quantification_simple -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification -U -t agreement -n quantification -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification_complex -U -t agreement -n quantification_complex -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/combination_simple -U -t agreement -n combination_simple -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/combination -U -t agreement -n combination -i 100 -M -H -N --config-values --world-size 100 --captions-per-instance 5

# classification
rm -r examples/classification
python generate.py -d examples/classification/oneshape -U -t classification -n oneshape -i 100 -M -H -N
python generate.py -d examples/classification/multishape -U -t classification -n multishape -i 100 -M -H -N
python generate.py -d examples/classification/countshape -U -t classification -n countshape -i 100 -M -H -N

# readme data
rm -r examples/readme
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n multishape -s 3,2,1 -i 100 -M -T
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n spatial -s 3,2,1 -i 100 -M
