#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Create a set of instances from the parameters given in a CVS file

Inputs:
- Input file name with parameters of instances
- Output file name with description of the corresponding instances
- Directory name of the instance files

Outputs
- File with description of instances
- Set of instances in the directory

See: 
puboi_param.csv and parameters_generator_evoCOP22.R for CSV format


Examples to execute this code from instance directory:
python ../src/generator/make_instances.py -I small/puboi_param_small.csv -O small/puboi_description_small.csv -D small
python ../src/generator/make_instances.py -I puboi_param.csv -O puboi_description.csv -D instancesEvoCOP

Becareful, the second example could be very long to execute...

Please cite this article if you use this code:
  Sara Tari, Sebastien Verel, and Mahmoud Omidvar. 
  "PUBOi: A Tunable Benchmark with Variable Importance." 
  In European Conference on Evolutionary Computation in Combinatorial Optimization (Part of EvoStar), pp. 175-190. Springer, Cham, 2022.

@authors: LEC group, LISIC Lab, Univ. of Littoral Opal Coast, France

"""
import argparse
import os
import puboi_generator as wg
import pandas as pd


def make_instances():


    parser = argparse.ArgumentParser(description='Process input file name, output file name, and output directory name of instances.')
    parser.add_argument("-I2", type=str, help='input file name with parameters of instances deg2', default="puboi_param_1000seed_deg2.csv")
    parser.add_argument("-I10", type=str, help='input file name with parameters of instances deg10', default="puboi_param_1000seed_deg10.csv")
    parser.add_argument("-D2", type=str, help='output directory name for saving the instance files deg2.', default="instances/deg2")
    parser.add_argument("-D10", type=str, help='output directory name for saving the instance files deg10.', default="instances/deg10")
    args = parser.parse_args()

    # Read the parameters for each case
    params_deg2 = pd.read_csv(args.I2, delimiter=' ')
    params_deg10 = pd.read_csv(args.I10, delimiter=' ')

    # Create directories for output if they don't exist
    os.makedirs(args.D2, exist_ok=True)
    os.makedirs(args.D10, exist_ok=True)

    # Starting IDs for deg2 and deg10
   # start_id_deg2 = 1000  # Starting ID for deg2 instances
    #start_id_deg10 = 1050  # Starting ID for deg10 instances
    # Generate valid instances for each case
    generate_valid_instances(params_deg2, args.D2, 30, args.I2)
    generate_valid_instances(params_deg10, args.D10, 30, args.I10)

	
def generate_valid_instances(params_df, output_prefix, num_required, input_file):
    valid_instance_count = 0
    total_attempts = 0
    idx = 0
    max_attempts = 10000  # Increase if necessary
    #id_instance = start_id
    valid = False
    while valid_instance_count < num_required and total_attempts < max_attempts:
        # Get the parameters for the current instance
        row = params_df.iloc[idx % len(params_df)]
        id_instance = int(row['id'])

        # Extract parameters
        n = int(row['n'])
        importance = {
            'size': [int(s) for s in row['size'].split(",")],
            'degree': [float(di) for di in row['degree'].split(",")]
        }
        m = int(row['m'])
        p_function = [float(pi) for pi in row['p'].split(',')]
        factor = float(row['factor'])
        shift = bool(int(row['shift']))
        seed = int(row['seed']) #+ total_attempts  # Modify the seed to ensure uniqueness
        typeWeight = int(row['typeWeight'])

        # Build portfolio
        builder = wg.PortfolioBuilder(n)
        portfolio = builder.make()

        # Initialize generator with the modified seed
        generator = wg.PUBOi_generator(
            id_instance=id_instance,
            m=m,
            importance=importance,
            factor=factor,
            portfolio=portfolio,
            p_function=p_function,
            typeWeight=typeWeight,
            shift=shift,
            seed=seed,  # Use the modified seed
            input_file=input_file,
            output_dir=output_prefix
        )
		# print parameters
        print_generator(id_instance, generator)
        # Generate function
        W = generator.make()
        
        if W is not None:
 
            filenameOut = f"{output_prefix}/puboi_{id_instance}.json"
            W.to_json(filenameOut, generator)
            print(f"Valid instance saved: {filenameOut}")
            valid_instance_count += 1
            valid = True
           # id_instance += 1
           
        else:
            print(f"Invalid instance for id {id_instance} with seed {seed}, regenerating.")
            valid = False
           # break
        total_attempts += 1
        if valid:
            idx += 1  # Move to next parameter set
        # Do not increment idx here, since we want to keep using the same parameters but different seeds

    if valid_instance_count < num_required:
        print(f"Could not generate {num_required} valid instances within {max_attempts} attempts.")

    return
# def make_instances():
# 	parser = argparse.ArgumentParser(description = 'Process input file name, output file name, and output directory name of instances.')
# 	parser.add_argument("-I", type = str, help = 'input file name with parameters of instances', default = "puboi_param_1000seed.csv")
# 	parser.add_argument("-O", type = str, help = 'output file name with description of the corresponding instances', default = "puboi_description_1000seed.csv")
# 	parser.add_argument("-D", type = str, help = 'output directory name for saving the instance files.', default = "../../instances/small/")
# 	args = parser.parse_args()
# 	input_name   = args.I
# 	output_name  = args.O
# 	dir_instance = args.D

# 	# read file with main parameters of instances
# 	df = pd.read_csv(input_name, delimiter = ' ')

# 	# Output file of instance description: parameters header
# 	f = open(output_name, "w")

# 	f.write("id type n n_class size degree factor typeWeight n_p p m shift seed bound\n")

# 	# Classes of importance
# 	importance = { }

# 	factor = 1.0
# 	seed   = 0

# 	for idx, row in df.iterrows():
# 		id_instance = int(row['id'])

# 		n = int(row['n'])

# 		importance['size'] = [ int(s) for s in row['size'].split(",") ]
# 		importance['degree'] = [ float(di) for di in row['degree'].split(",") ]

# 		m = int(row['m'])

# 		p_function = [ float(pi) for pi in row['p'].split(',') ]

# 		factor = float(row['factor'])
# 		shift  = bool(row['shift'])
# 		seed   = int(row['seed'])

# 		# 4 functions in the portfolio (cf. )
# 		builder   = wg.PortfolioBuilder(n)
# 		portfolio = builder.make()

# 		# generator
# 		generator = wg.PUBOi_generator(id_instance=id_instance, m = m, importance = importance, factor = factor, portfolio = portfolio, 
# 			                           p_function = p_function, typeWeight = 0, shift = shift, seed = seed)
		
# 		# print parameters
# 		print_generator(id_instance, generator)

# 		# Create function
# 		W = generator.make()

# 		# print parameters into file
# 		write_generator(id_instance, generator, f)

# 		# export to json file
# 		filenameOut = ""
# 		if len(dir_instance) > 0:
# 			filenameOut = dir_instance + "/"
# 		filenameOut = filenameOut + "puboi_" + str(id_instance) + ".json"

# 		W.to_json(filenameOut, generator)
# 		#print(importance['degree'])
# 	f.close()

def print_generator(id_instance, generator):
	# print parameters
	print(str(id_instance), end='')
	print(" puboi", end='')
	print(" " + str(generator.n) + " %d \"%d" % (generator.n_class, generator.importance['size'][0]), end='')
	for i in range(1, generator.n_class):
		print(',' + str(generator.importance['size'][i]), end='')
	print('\" \"%.6f' % generator.importance['degree'][0], end='')
	for i in range(1, generator.n_class):
		print(',%.6f' % generator.importance['degree'][i], end='')
	print(('\" %.6f' % generator.factor) + ' ' + str(generator.typeWeight), end='')
	print(" 4 \"%.6f,%.6f,%.6f,%.6f\"" % (generator.p_function[0], generator.p_function[1], generator.p_function[2], generator.p_function[3]), end='')
	print(' ' + str(generator.m), end='')
	if generator.shift:
		print(' 1', end='')
	else:
		print(' 0', end='')
	print(' ' + str(generator.seed), end='')

	if hasattr(generator, 'bound'):
		print(' ' + str(generator.bound))
	else:
		print('')

def write_generator(id_instance, generator, f):
	# print parameters
	f.write(str(id_instance))
	f.write(" puboi")
	f.write(" " + str(generator.n) + " %d \"%d" % (generator.n_class, generator.importance['size'][0]))
	for i in range(1, generator.n_class):
		f.write(',' + str(generator.importance['size'][i]))
	f.write('\" \"%.6f' % generator.importance['degree'][0])
	for i in range(1, generator.n_class):
		f.write(',%.6f' % generator.importance['degree'][i])
	f.write(('\" %.6f' % generator.factor) + ' ' + str(generator.typeWeight))
	f.write(" 4 \"%.6f,%.6f,%.6f,%.6f\"" % (generator.p_function[0], generator.p_function[1], generator.p_function[2], generator.p_function[3]))
	f.write(' ' + str(generator.m))
	if generator.shift:
		f.write(' 1')
	else:
		f.write(' 0')
	f.write(' ' + str(generator.seed))

	if hasattr(generator, 'bound'):
		f.write(' ' + str(generator.bound))
	f.write('\n')

if __name__ == "__main__":
	make_instances()