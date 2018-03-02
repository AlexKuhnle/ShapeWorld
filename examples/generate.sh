# agreement
rm -r examples/agreement
python generate.py -d examples/agreement/oneshape -U -t agreement -n oneshape -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/multishape -U -t agreement -n multishape -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/spatial -U -t agreement -n spatial -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/relational -U -t agreement -n relational -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification_simple -U -t agreement -n quantification_simple -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification -U -t agreement -n quantification -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/quantification_complex -U -t agreement -n quantification_complex -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5
python generate.py -d examples/agreement/combination -U -t agreement -n combination -M -H -i 100 --config-values --world-size 100 --captions-per-instance 5

# classification
rm -r examples/classification
python generate.py -d examples/classification/oneshape -U -t classification -n oneshape -M -H -i 100
python generate.py -d examples/classification/multishape -U -t classification -n multishape -M -H -i 100
python generate.py -d examples/classification/countshape -U -t classification -n countshape -M -H -i 100

# readme data
rm -r examples/readme
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n multishape -s 3,2,1 -i 100 -M -T
python generate.py -d examples/readme -a tar:bzip2 -t agreement -n spatial -s 3,2,1 -i 100 -M
