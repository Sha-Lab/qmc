
run_main:
	python exp_utils/run.py main.py $(p)

run_main-d:
	python exp_utils/run.py main.py $(p) -d

run_arqmc:
	python exp_utils/run.py arqmc.py $(p)

run_arqmc-d:
	python exp_utils/run.py arqmc.py $(p) -d
